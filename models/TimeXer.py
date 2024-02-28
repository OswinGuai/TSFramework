import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import  Decoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted,  PositionalEmbedding
import numpy as np

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class TimeXer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512, dropout=0.5, activation='gelu', output_attention=False,
                 device=0, stage = 'train',use_norm = 1,patch_len=16,timestamp_feature='h',
                 task_name='Forecast', features='M',embed = 'timeF',freq='h',factor=1,**kwargs):
        super(TimeXer, self).__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.device = device
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.stage = stage
        self.d_model = d_model
        self.use_norm = use_norm
        self.patch_len = patch_len
        self.target_num = 1 if features == 'MS' else enc_in
        self.cross_mask = False if features == 'MS' else True
        self.patch_num = int(self.seq_len // patch_len)
        # Embedding
        self.patch_embedding = PatchEmbedding(
            self.target_num, self.d_model, self.patch_len, dropout
        )
        self.co_embedding = DataEmbedding_inverted(
            self.seq_len, self.d_model, embed, freq, dropout
        )

        # Decoder-only architecture
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model,n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model,n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation = activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(enc_in,self.head_nf, pred_len, head_dropout=dropout)

   

    def get_mask(self, L, mark):
        if mark is not None:
            mask = torch.eye(L+4,dtype=torch.bool)
            mask = mask[:-4,:]
        else:
            mask = torch.eye(L, dtype=torch.bool)
        return mask


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out, n_vars = self.patch_embedding(x_enc[:,:,-1].unsqueeze(-1).permute(0,2,1))
        enc_out_glb = self.co_embedding(x_enc[:,:,-1].unsqueeze(-1), None)
        enc_out = torch.cat([enc_out, enc_out_glb], dim=1)
        co_enc_out = self.co_embedding(x_enc[:,:,:-1], x_mark_enc)

        dec_out = self.decoder(enc_out, co_enc_out)
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        dec_out = dec_out.permute(0, 1, 3, 2)
        # Decoder
        # import pdb
        # pdb.set_trace()
        dec_out = self.head(dec_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_covariate(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out, n_vars = self.patch_embedding(x_enc.permute(0,2,1))
        enc_out_glb = self.co_embedding(x_enc, None)
        enc_out_glb = torch.reshape(enc_out_glb, (enc_out_glb.shape[0] * enc_out_glb.shape[1], 1, enc_out_glb.shape[2]))
        enc_out = torch.cat([enc_out, enc_out_glb], dim=1)
        co_enc_out = self.co_embedding(x_enc, x_mark_enc)

        dec_out = self.decoder(enc_out, co_enc_out)
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        dec_out = dec_out.permute(0, 1, 3, 2)
        # Decoder
        dec_out = self.head(dec_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # if self.task_name == 'long_term_forecast_covariate':
        #     out = []
        #     for i in range(x_enc.shape[-1]):
        #         if i > 0:
        #             x_in = torch.cat(
        #                 [x_enc[:, :, :i], x_enc[:, :, i + 1:], x_enc[:, :, i].unsqueeze(-1)], dim=-1)
        #             dec_out = self.forecast(x_in, x_mark_enc, x_dec, x_mark_dec)
        #             out.append(dec_out.squeeze(-1))
        #         else:
        #             x_in = torch.cat([x_enc[:, :, i + 1:], x_enc[:, :, i].unsqueeze(-1)], dim=-1)
        #             dec_out = self.forecast(x_in, x_mark_enc, x_dec, x_mark_dec)
        #             out.append(dec_out.squeeze(-1))
        #     output = torch.stack(out, dim=-1)
        #     return output  # [B, L, D]

        if self.task_name == 'long_term_forecast_covariate':
            dec_out = self.forecast_covariate(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]