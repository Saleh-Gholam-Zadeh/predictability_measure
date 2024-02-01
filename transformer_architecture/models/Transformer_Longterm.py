import time

from torch.nn.modules import activation
import torch
import torch.nn as nn
from dataclasses import dataclass

from transformer_architecture.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from transformer_architecture.layers.SelfAttention_Family import FullAttention, AttentionLayer
from transformer_architecture.layers.Embed import DataEmbedding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@dataclass
class TSConfig:

    # block_size: int = 128   #0.0614 seconds context
    # vocab_size: int = 25   # in TS it is input dimension e.g 25 features // token embedding layer was changed to simple linear layer
    # n_layer: int = 6
    # n_head: int = 6
    # n_embed: int = 384 #C #every head is 768/12= 64 dimensional  we assumed also head_size=C= n_embed
    # head_size = n_embed//n_head #C = 64
    # dropout: float = 0.1
    #-----------------------------------
    enc_in:  int = 1 # number of channels(features) in the input #would be replaced by number of features in the input
    dec_in:  int = 1 # number of input channels(features) in the decoder #would be replaced by number of features in the input
    c_out: int =  1 # output dim. seems to be always the same as dec_in except for the classification schema
    #d_model: int = 512 #n_embed originally=512
    d_model: int = 512
    
    #dropout: float = 0.1
    dropout: float = 0.1

    pred_len: int = 75
    output_attention = True  # whether to output attention in encoder
    
    n_heads: int = 8
    activation: str = 'gelu'
    e_layers: int = 2
    d_layers: int = 1

    embed = None
    freq =  None
    d_ff = None # here none will be translate to 4*d_model
    factor = None
    #n_embed: int = 384/// hamun d_model ast #C #every head is 768/12= 64 dimensional  we assumed also head_size=C= n_embed
    #head_size = n_embed//n_head #C = 64
    
class LongTermModel(nn.Module):
  """
  Vanilla Transformer
  """

  def __init__(self, configs):
      super(LongTermModel, self).__init__()
      self.pred_len = configs.pred_len
      self.output_attention = configs.output_attention

      # Embedding
      self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                          configs.dropout)
      self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                          configs.dropout)
      # Encoder
      self.encoder = Encoder(
          [
              EncoderLayer(
                  AttentionLayer(
                      FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                  configs.d_model,
                  configs.d_ff,
                  dropout=configs.dropout,
                  activation=configs.activation
              ) for l in range(configs.e_layers)
          ],
          norm_layer=torch.nn.LayerNorm(configs.d_model)
      )
      # Decoder
      self.decoder = Decoder(
          [
              DecoderLayer(
                  AttentionLayer(
                      FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                      configs.d_model, configs.n_heads),
                  AttentionLayer(
                      FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                      configs.d_model, configs.n_heads),
                  configs.d_model,
                  configs.d_ff,
                  dropout=configs.dropout,
                  activation=configs.activation,
              )
              for l in range(configs.d_layers)
          ],
          norm_layer=torch.nn.LayerNorm(configs.d_model),
          projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
      )

  def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

      '''
      x_enc: context time_window which is fed to the encoder
      x_mark_enc: usually empty and with no effect
      x_dec: if None --> [B,S//2:S,features]
      x_mark_dec: zeros(B,pred_len,features)  ---> will be concatenated with the x_dec and will be embedded together
      '''

      now = time.time()
      #print('Transformer forward-pass is started...')

      B,T,C = x_enc.shape
      #print('B,T,C =',B,T,C)
      enc_out = self.enc_embedding(x_enc, x_mark_enc)
      enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
      #print('Encoder is finished...')

      if x_dec is None:
        x_dec = x_enc[:,:,:]   # x_dec = x_enc[:,T//2:,:] --> original implementation
        #print('line 104 transformer_Longter.py   before concatenation  x_dec.shape=',x_dec.shape)

      if x_mark_dec is None:
        x_mark_dec = torch.zeros(B,self.pred_len,C).to(device)
        #print("x_mark_dec.shape", x_mark_dec.shape )

      dec_out = self.dec_embedding(x_dec, x_mark_dec)
      dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
      #print('Transformer forward-pass took: ',time.time()-now)
      if self.output_attention:
          return dec_out[:, -self.pred_len:, :], attns
      else:
          return dec_out[:, -self.pred_len:, :]  # [B, L, D]

if __name__ == "__main__":
  inp_enc = torch.randn(10,50,7).to(device)
  my_conf = TSConfig(enc_in=inp_enc.shape[-1], dec_in=inp_enc.shape[-1])
  print(my_conf)
  ts_model = LongTermModel(my_conf).to(device)
  print(ts_model)
  inp_enc = torch.randn(10,50,7).to(device)
  inp_dec = torch.randn(10,10,7).to(device)
  mark_dec = torch.zeros((10,30,1)).to(device)
  a,b = ts_model(inp_enc,None,inp_dec,None)
  print(a.shape)
  print("Attention weights:",b[0].shape , b[1].shape)