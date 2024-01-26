import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        #x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
    
class DataEmbedding_linear(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_linear, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        # x = self.value_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_linear_noPosition(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_linear_noPosition, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_inverted_new(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted_new, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # first conv along time axis.
        # then back, meaning obvious features along time axis
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TokenTCNEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenTCNEmbedding, self).__init__()
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,kernel_size=3, padding=padding, padding_mode='circular',bias=False)
        self.tokenConv = TemporalConvNet(num_inputs=1, num_channels=[512] * 5, kernel_size=3)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.embed = Embed(minute_size, d_model)
        elif freq == 'h':
            self.embed = Embed(hour_size, d_model)
        elif freq == 'w':
            self.embed = Embed(weekday_size, d_model)
        elif freq == 'd':
            self.embed = Embed(day_size, d_model)
        elif freq == 'm':
            self.embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        x_emb = self.embed(x[:, :, 0])
        return x_emb


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map = {'h': 3, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DateEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DateEmbedding, self).__init__()

        self.PI = 3.1415
        self.embed = nn.Linear(2, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.sin_t = nn.Parameter(torch.empty(1))
        self.cos_t = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.sin_t)
        nn.init.uniform_(self.cos_t)

    def cycle_time(self, m, d):
        # scale = self.scales[scale_key]
        # For continuity and prediority
        t_input = (m + d / 32) / 13 * self.PI * 2
        t_sin = (1 + torch.sin(t_input * self.sin_t)) / 2
        t_cos = (1 + torch.cos(t_input * self.cos_t)) / 2
        return 10 * torch.cat([t_sin, t_cos], -1)
        # return torch.cat([t_sin,t_cos],-1)

    def forward(self, m, d):
        x_emb = self.cycle_time(m, d).float()
        x_emb = self.embed(x_emb)
        return self.dropout(x_emb)


class TimeEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TimeEmbedding, self).__init__()

        self.PI = 3.1415
        self.embed = nn.Linear(2, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def cycle_time(self, h, m):
        # scale = self.scales[scale_key]
        # For continuity and prediority
        t_input = (h + m / 60) / 24 * self.PI * 2
        t_sin = (1 + torch.sin(t_input)) / 2
        t_cos = (1 + torch.cos(t_input)) / 2
        return torch.cat([t_sin, t_cos], -1)

    def forward(self, m, d):
        x_emb = self.cycle_time(m, d).float()
        x_emb = self.embed(x_emb)
        return self.dropout(x_emb)


class CycleTimeEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(CycleTimeEmbedding, self).__init__()

        self.PI = 3.1415
        self.start = {'h': 0, 'm': 1, 'd': 1}
        self.scales = {'h': 24, 'm': 13, 'd': 32}
        self.embed = nn.Linear(2, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        #self.sin_t = nn.Parameter(torch.empty(1))
        #self.cos_t = nn.Parameter(torch.empty(1))
        #nn.init.uniform_(self.sin_t)
        #nn.init.uniform_(self.cos_t)

    def cycle_time(self, t, scale_key):
        scale = self.scales[scale_key]
        t_input = t / scale * self.PI * 2
        #t_sin = (1 + torch.sin(t_input * self.sin_t)) / 2
        #t_cos = (1 + torch.cos(t_input * self.cos_t)) / 2
        t_sin = (1 + torch.sin(t_input)) / 2
        t_cos = (1 + torch.cos(t_input)) / 2
        return torch.cat([t_sin, t_cos], -1)

    def forward(self, x, scale_key='m'):
        x_emb = self.cycle_time(x, scale_key).float()
        x_emb = self.embed(x_emb)

        return self.dropout(x_emb)




