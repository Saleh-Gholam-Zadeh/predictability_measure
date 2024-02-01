import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):  #AttentionHead
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape #B,L,H,E. : batch, time_length_query, num_heads, embeded_features_q 
                            # KEYS: B,S,H,E
        _ , S, _, D = values.shape #B,S,H,D. : batch, time_length_values, num_heads, embeded_features_val
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # B,H,L,S:  batch, num_heads, time_length_values,  time_length_query
        if self.mask_flag:
            #print('masking with ...')
            if attn_mask is None:
              #print('TriangularCausalMask')
              attn_mask = TriangularCausalMask(B, L, device=queries.device)
            #print('attn_mask:',attn_mask)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        else:
            pass
          #print('attention without masking')

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module): #MultiHeadAttention

  def __init__(self,attention , d_model, n_heads, d_keys=None, d_values=None):
    super().__init__()

    d_keys = d_keys or (d_model // n_heads)
    d_values = d_values or (d_model // n_heads)

    self.inner_attention = attention
    self.query_projection = nn.Linear(d_model, d_keys * n_heads)   #  query generation
    self.key_projection = nn.Linear(d_model, d_keys * n_heads)     #  Key generation
    self.value_projection = nn.Linear(d_model, d_values * n_heads) #  Val generation
    self.out_projection = nn.Linear(d_values * n_heads, d_model)
    self.n_heads = n_heads


  def forward(self, queries, keys, values, attn_mask):
    '''
    query:[B,L,d_model]
    key  :[B,L,d_model]
    val  :[B,L,d_model]

    output: [B,L,d_model]
    '''

    B, L, _ = queries.shape # B,L,d_model//num_heads
    _, S, _ = keys.shape
    H = self.n_heads

    queries = self.query_projection(queries).view(B, L, H, -1) # B,L,H,d_model//H.     H=num_heads
    keys = self.key_projection(keys).view(B, S, H, -1)
    values = self.value_projection(values).view(B, S, H, -1)

    out, attn = self.inner_attention(
        queries,
        keys,
        values,
        attn_mask
    )
    out = out.view(B, L, -1) #B,L,d_model

    return self.out_projection(out), attn

if __name__ == '__main__':
  my_att = FullAttention(output_attention=True)
  my_att_layer = AttentionLayer(my_att,16,4) # #d_model=16 ,num_head=4
  x= torch.randn(2,10,16)
  print(my_att_layer(x,x,x,None)[0].shape) #torch.Size([2, 10, 16])

  print(my_att_layer(x,x,x,None)[1].shape) #torch.Size([2, 4, 10, 10])