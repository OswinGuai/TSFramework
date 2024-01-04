from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import numpy as np

from models.modules.decoder import Decoder, DecoderLayer
from models.modules.encoder import Encoder, EncoderLayer

from models.modules.attn import FullAttention, AttentionLayer
from models.modules.embed import PatchEmbedding


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


class PatchTST(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, factor, stride=8, patch_len=16, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512, dropout=0.5, activation='gelu', output_attention=False,
                 device=0, stage='train',**kwargs):
        super(PatchTST, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.device = device
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.stage = stage


        # Encoder
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, stride, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
            )

        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)

        # self.target_projector = nn.Linear(self.enc_in, self.c_out, bias=True)
        self.target_projector = nn.Linear(self.enc_in, 1, bias=True)

        self.global_step = 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape # B L N

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.stage=='train':
            dec_out = self.target_projector(dec_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:,:,:self.c_out]

