
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
#import math
from transformer_architecture.layers.Embed import DataEmbedding


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask



class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False):  # It does the calculation of all Heads, but queries and keys and vals are generated in attention layer. this class is used
        # in the middle of the attentionlayer
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):

        B, L, H, E = queries.shape  #: batch, time_length_query, num_heads, embeded_features_q
        # B, S, H, E =    KEYS:
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B *1 *1 *1
        delta = 1.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B *1 *1 *1

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries,
                              keys) * tau + delta  # B,H,L,S:  batch, num_heads, time_length_values,  time_length_query

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), _)


class AttentionLayer(nn.Module):

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        '''
        query:[B,L,d_model]
        key  :[B,L,d_model]
        val  :[B,L,d_model]

        output: [B,L,d_model]
        '''
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # print("attn_mask: ", attn_mask)
        #
        # print('B,L,H: ', B, L, H)
        # print("before generation: queries.shape: ",
        #       queries.shape)  # (B,L,d_m) --LINEAR_layer--> (B,L,d_m) Then in the next line --reshape--> (B,L,n_heads,d_m//n_heads)
        queries = self.query_projection(queries).view(B, L, H,
                                                      -1)  # query genrearation. (B,L,d_m) -linear-> (B,L,d_m) and then --.view-->    (B,L,H,d_m//H) . let's call d_m//H as e. then 'blhe' * 'bshe' -> 'bhls' this operation also ensures that calcaulations are separated among different heads
        #print("queries.shape: ", queries.shape)
        keys = self.key_projection(keys).view(B, S, H, -1)  # key genrearation
        #print("keys.shape:    ", keys.shape)
        values = self.value_projection(values).view(B, S, H, -1)  # value genrearation
        #print("values.shape:  ", values.shape)

        # applying  attention mechanism
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau, delta
        )
        out = out.view(B, L, -1)  # "B,L,H,d_val(=d_model//H)" --> "B,L,H*d_moded//H"

        return self.out_projection(out), attn


if __name__ == "__main__":
    b = 2
    l = 20
    d = 8
    h = 4
    features = 1
    x = torch.randn(b, l, features)
    print("x.shape: ", x.shape)
    saleh_emb = DataEmbedding(c_in=features, d_model=d)
    cur_x = saleh_emb(x)
    print("cur_x.shape:  ", cur_x.shape)

    saleh_att = DSAttention()
    saleh_att_layer = AttentionLayer(saleh_att, d_model=d, n_heads=h)
    saleh_mask = TriangularCausalMask(B=b, L=l)
    output_1, _ = saleh_att_layer(cur_x, cur_x, cur_x, attn_mask=saleh_mask)
    print("output_1.shape:", output_1.shape)
