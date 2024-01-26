import torch
import torch.nn as nn
import numpy as np

from models.modules.decoder import Decoder, DecoderLayer
from models.modules.encoder import Encoder, EncoderLayer

from models.modules.attn import FullAttention, AttentionLayer
from models.modules.embed import DataEmbedding, DateEmbedding, CycleTimeEmbedding


class Transformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512, dropout=0.5, activation='gelu', output_attention=False,
                 device=0, timestamp_feature='h', **kwargs):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.device = device
        self.output_attention = output_attention
        self.c_out = c_out
        self.timestamp_feature = timestamp_feature

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
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

        enc_out = self.enc_embedding(x_enc)
        if self.timestamp_feature != 'none':
            hour_enc = self.hour_embedding(x_mark_enc, scale_key='h')
            enc_out = enc_out + hour_enc

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        x_dec_input = x_dec
        dec_out = self.dec_embedding(x_dec_input)
        if self.timestamp_feature != 'none':
            hour_dec = self.hour_embedding(x_mark_dec, scale_key='h')
            dec_out = dec_out + hour_dec

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        outputs = self.projection_a(dec_out)

        self.global_step = self.global_step + 1

        return outputs[:, -self.pred_len:, :]
