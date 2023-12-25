import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.embed import DataEmbedding, DateEmbedding, CycleTimeEmbedding


class LSTM(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model=512, dropout=0.0, embed='fixed', 
                 device=torch.device('cuda:0'), **kwargs):
        super(LSTM, self).__init__()
        self.pred_len = pred_len
        self.hidden_size = d_model
        self.init = nn.Conv1d(in_channels=enc_in,
                              out_channels=self.hidden_size,
                              kernel_size=3,
                              stride=1,
                              padding=2,
                              padding_mode='circular',
                              bias=False)
        self.LSTM = torch.nn.LSTM(input_size=enc_in, hidden_size=self.hidden_size, num_layers=1,
                                  batch_first=True)
        #self.enc_embedding = DataEmbedding(enc_in, d_model, embed, 'h', dropout)
        #self.dec_embedding = DataEmbedding(dec_in, d_model, embed, 'h', dropout)
        self.projection_a = nn.Linear(d_model, c_out, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # enc_out = self.init(x_enc.permute(0, 2, 1)).transpose(1, 2)
        enc_out = x_enc#self.enc_embedding(x_enc, x_mark_enc)
        dec_out = x_dec#self.dec_embedding(x_dec, x_mark_dec)
        output, (h, c) = self.LSTM(torch.cat((enc_out,dec_out[:,-self.pred_len:,:]),dim=1))
        outputs = self.projection_a(output)[:, -self.pred_len:, :]

        return outputs[:, -self.pred_len:, :]


