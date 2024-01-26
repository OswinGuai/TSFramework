import torch
import torch.nn as nn
import numpy as np

from models.modules.decoder import Decoder, DecoderLayer
from models.modules.encoder import Encoder, EncoderLayer

from models.modules.attn import FullAttention, AttentionLayer
from models.modules.embed import DataEmbedding, DateEmbedding, CycleTimeEmbedding,DataEmbedding_inverted,DataEmbedding_linear,DataEmbedding_linear_noPosition


class Transformer_new(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512, dropout=0.05, activation='gelu', output_attention=False,
                 device=0, timestamp_feature='h', **kwargs):
        super(Transformer_new, self).__init__()
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.device = device
        self.output_attention = output_attention
        self.c_out = c_out
        self.timestamp_feature = timestamp_feature

        # * Original Encoder Embedding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.enc_embedding = nn.ModuleList([DataEmbedding(enc_in,d_model,dropout) for i in range(self.seq_len)])
        
        # * Multi Encoder Embedding Invered
        # self.enc_embedding = nn.ModuleList([DataEmbedding_inverted(seq_len, d_model, 'timeF', 'h', dropout) for i in range(self.enc_in)])

        # * Single Encoder Embedding Inverted
        # self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, 'timeF', 'h', dropout)

        # * Single Encoder Embedding Linear
        # self.enc_embedding = DataEmbedding_linear(enc_in,d_model,dropout)
        # self.enc_embedding = DataEmbedding_linear_noPosition(enc_in,d_model,dropout)

        # * Multi Encoder Embedding Linear
        # self.enc_embedding = nn.ModuleList([DataEmbedding_linear(enc_in,d_model,dropout) for i in range(self.seq_len)])
        # self.enc_embedding = nn.ModuleList([DataEmbedding_linear_noPosition(enc_in,d_model,dropout) for i in range(self.seq_len)])

        # * Original Decoder Embedding
        # self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        self.dec_embedding = nn.ModuleList([DataEmbedding(dec_in,d_model,dropout) for i in range(self.label_len+self.pred_len)])

        # * Multi Decoder Embedding Inverted
        # self.dec_embedding = nn.ModuleList([DataEmbedding_inverted(seq_len+label_len, d_model, 'timeF', 'h', dropout) for i in range(self.dec_in)])

        # * Single Decoder Embedding Inverted
        # self.dec_embedding = DataEmbedding_inverted(seq_len+label_len, d_model, 'timeF', 'h', dropout)

        # * Single Decoder Embedding Linear
        # self.dec_embedding = DataEmbedding_linear(dec_in,d_model,dropout)

        # * Multi Decoder Embedding Linear
        # self.dec_embedding = nn.ModuleList([DataEmbedding_linear(dec_in,d_model,dropout) for i in range(self.label_len+self.pred_len)])
        # self.dec_embedding = nn.ModuleList([DataEmbedding_linear_noPosition(dec_in,d_model,dropout) for i in range(self.label_len+self.pred_len)])
        


        self.hour_embedding = CycleTimeEmbedding(int(d_model), dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection_a = nn.Linear(d_model, self.c_out, bias=True)

        self.global_step = 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, bp_bit=True):

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        if self.timestamp_feature != 'none':
            x_mark_enc = None

        # * Original Encoder Embedding | Single Encoder Embedding Linear
        # enc_out = self.enc_embedding(x_enc)
        
        # * Single Encoder Embedding Inverted
        # enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        
        # * Multi Encoder Embedding Inverted
        # enc_out_single_array=[]
        # for index in range(self.enc_in):
        #     x_enc_single = x_enc[:,:,index].unsqueeze(2)
        #     enc_out_single = self.enc_embedding[index](x_enc_single,x_mark_enc)
        #     enc_out_single_array.append(enc_out_single)
        # enc_out = torch.cat(enc_out_single_array,dim=1)

        # * Multi Encoder Embedding Linear
        enc_out_single_array = []
        for index in range(self.seq_len):
            x_enc_single = x_enc[:,index,:].unsqueeze(1)
            enc_out_single = self.enc_embedding[index](x_enc_single)
            enc_out_single_array.append(enc_out_single)
        enc_out = torch.cat(enc_out_single_array,dim=1)

        if self.timestamp_feature != 'none':
            hour_enc = self.hour_embedding(x_mark_enc, scale_key='h')
            enc_out = enc_out + hour_enc

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # import pdb
        # pdb.set_trace()

        x_dec_input = x_dec

        # * Original Decoder Embedding | Single Decoder Embedding Linear
        # dec_out = self.dec_embedding(x_dec_input)

        # * Single Decoder Embedding Inverted
        # dec_out = self.dec_embedding(x_dec, x_mark_dec) 

        # * Multi Decoder Embedding Inverted
        # dec_out_single_array=[]
        # for index in range(self.dec_in):
        #     x_dec_single = x_dec[:,:,index].unsqueeze(2)
        #     dec_out_single = self.dec_embedding[index](x_dec_single,x_mark_dec)
        #     dec_out_single_array.append(dec_out_single)
        # dec_out = torch.cat(dec_out_single_array,dim=1)

        # * Multi Decoder Embedding Linear
        dec_out_single_array = []
        for index in range(self.label_len+self.pred_len):
            x_dec_single = x_dec[:,index,:].unsqueeze(1)
            dec_out_single = self.dec_embedding[index](x_dec_single)
            dec_out_single_array.append(dec_out_single)
        dec_out = torch.cat(dec_out_single_array,dim=1)


        if self.timestamp_feature != 'none':
            hour_dec = self.hour_embedding(x_mark_dec, scale_key='h')
            dec_out = dec_out + hour_dec

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        outputs = self.projection_a(dec_out)

        self.global_step = self.global_step + 1

        return outputs[:, -self.pred_len:, :]

