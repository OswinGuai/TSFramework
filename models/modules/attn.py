import torch
import torch.nn as nn

import numpy as np
from math import sqrt


class TriangularCausalMask2:
    def __init__(self, B, L, H, device="cpu"):
        mask_shape = [B, H, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class SparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SparseAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        '''queries = queries.permute(1, 0, 2, 3)
        keys = keys.permute(1, 0, 2, 3)
        values=values.permute(1,0,2,3)'''

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        '''scores = torch.zeros([B, H, L, S]).to(queries.device)

        for i in range(L):

            #local attention
            for u in range(int (max((math.log(i*1.0+1,2.0)),1))):
                scores[:, :, i, i - u] = torch.einsum("bhe,bhe->bh", queries[:, i, :, :], keys[:, i - u, :, :])
            u = 1

            # sparse attention
            while(i-u>=0):
                scores[:, :, i, i - u] =torch.einsum("bhe,bhe->bh", queries[:, i, :, :] ,keys[:, i - u, :, :])
                u = u * 2'''
        # print(time.time()-t)
        if self.mask_flag:
            if True:

                attn_mask = TriangularCausalMask2(B, L, H, device=queries.device)

                attn_mask.mask[:, :, :, :] = True

                a = torch.ones([B, H, attn_mask.mask.shape[-1]], dtype=torch.bool).to(queries.device)
                u = -1
                while (abs(u) < attn_mask.mask.shape[-1]):
                    a[:, :, u] = False
                    u = u * 4
                '''for u in range((int)(math.log(attn_mask.mask.shape[-1]))):
                    a[:, :, -u] = False'''
                for i in range(attn_mask.mask.shape[-1]):
                    attn_mask.mask[:, :, i, :i + 1] = attn_mask.mask[:, :, i, :i + 1] & a[:, :, -i - 1:]
                    # attn_mask.mask[:, :, i, i] = False
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
