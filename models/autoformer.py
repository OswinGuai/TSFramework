import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.embed import DataEmbedding_value_embedding
from models.modules.embed import DataEmbedding, CycleTimeEmbedding
from models.layers.auto_correlation import AutoCorrelation, AutoCorrelationLayer
from models.layers.autoformer_layers import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=1, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, datetime_feature_setting='M_d_h', device=0, **kargs):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.attn = attn
        self.output_attention = output_attention
        self.datetime_feature_setting = datetime_feature_setting
        # decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        self.hour_embedding = CycleTimeEmbedding(int(d_model), dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, enc_in, bias=True)
        )
        projection_final=nn.Linear(enc_in, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        hour_enc = self.hour_embedding(x_mark_enc, scale_key='h')
        enc_out = enc_out + hour_enc
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        hour_dec = self.hour_embedding(x_mark_dec, scale_key='h')
        dec_out = dec_out + hour_dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        # enc
        enc_out = self.enc_embedding(x_enc)
        if self.datetime_feature_setting == 'M_d_h':
            hour_enc = self.hour_embedding(x_mark_enc[:, :, 2:3], scale_key='h')
            enc_out = enc_out + hour_enc

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        #dec_out = self.dec_embedding(seasonal_init)
        dec_out = self.dec_embedding(x_dec)
        if self.datetime_feature_setting == 'M_d_h':
            hour_dec = self.hour_embedding(x_mark_dec[:, :, 2:3], scale_key='h')
            dec_out = dec_out + hour_dec

        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        outputs = self.projection_final(dec_out)
        self.global_step = self.global_step + 1
        return outputs[:, -self.pred_len:, :-1], outputs[:, -self.pred_len:, -1:]

