import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
  def __init__(self, c_in):
    super(ConvLayer, self).__init__()
    self.downConv = nn.Conv1d(in_channels=c_in,
                              out_channels=c_in,
                              kernel_size=3,
                              padding=2,
                              padding_mode='circular') #with circular , input should have 3dim (have batch), although logically it is not necessary otherwise non-sense error would be raised. also along batch_dim (-3) no operation would happen
    self.norm = nn.BatchNorm1d(c_in)
    self.activation = nn.ELU()
    self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

  def forward(self, x):
    x = self.downConv(x.permute(0, 2, 1))
    x = self.norm(x)
    x = self.activation(x)
    x = self.maxPool(x)
    x = x.transpose(1, 2)
    return x

# cc = ConvLayer(16)
# cc(torch.randn(34,5,16)).shape.  #torch.Size([34, 4, 16])


class EncoderLayer(nn.Module):
  def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
    super(EncoderLayer, self).__init__()
    d_ff = d_ff or 4 * d_model
    self.attention = attention     #its an instance of AttentionLayer not instance of FullAttention, 
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
    self.activation = F.relu if activation == "relu" else F.gelu


  def forward(self, x, attn_mask=None):
    '''
    x:[B,L,d_model]. #attention layer  was already passed to the __init__
    
    out=[B,L,d_model]
    '''
    new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
    x = x + self.dropout(new_x)

    y = x = self.norm1(x)
    # print(x.shape)
    # print(y.shape)
    # print(y.transpose(-1, 1).shape)
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
    y = self.dropout(self.conv2(y).transpose(-1, 1))

    return self.norm2(x + y), attn

class Encoder(nn.Module):
  #def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
  def __init__(self, enc_layers, conv_layers=None, norm_layer=None):


    super(Encoder, self).__init__()
    self.enc_layers = nn.ModuleList(enc_layers)
    self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
    self.norm = norm_layer

  def forward(self, x, attn_mask=None):
    #print('Encder forward is called...')
    # x [B, L, D]
    attns = []
    if self.conv_layers is not None:
        print("self.conv_layers is not None:",self.conv_layers)
        for enc_layer, conv_layer in zip(self.enc_layers, self.conv_layers):
            x, attn = enc_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.enc_layers[-1](x)
        attns.append(attn)
    else:
        for enc_layer in self.enc_layers:
          #print("self.conv_layers is None:", self.conv_layers)
          #print('EncLayer is called inside Encoder forward:')
          x, attn = enc_layer(x, attn_mask=attn_mask)
          attns.append(attn)

    if self.norm is not None:
        x = self.norm(x)

    return x, attns

class DecoderLayer(nn.Module):
  def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
    super(DecoderLayer, self).__init__()
    d_ff = d_ff or 4 * d_model
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
    #print("line 103 Transformer_EncDecdropout inside the decoderlayer:",self.dropout)
    self.activation = F.relu if activation == "relu" else F.gelu

  def forward(self, x, cross, x_mask=None, cross_mask=None):
    '''
    x:[B,L,d_model] query from decoder #attention layer  was already passed to the __init__
    cross: [B,L,d_model] key and val from encoder
    '''
    #print('Decoder layer: attn_mask=',x_mask)
    x = x + self.dropout(self.self_attention(
        x, x, x,
        attn_mask=x_mask
    )[0])
    x = self.norm1(x)

    x = x + self.dropout(self.cross_attention(
        x, cross, cross,
        attn_mask=cross_mask
    )[0])
    y = x = self.norm2(x)
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
    y = self.dropout(self.conv2(y).transpose(-1, 1))

    return self.norm3(x + y) # note that it doesnt return attn_weight as well 

class Decoder(nn.Module):
  def __init__(self, layers, norm_layer=None, projection=None):
    '''
    layers is a list of decoder layers
    '''
    super(Decoder, self).__init__()
    self.layers = nn.ModuleList(layers)
    self.norm = norm_layer
    self.projection = projection

  def forward(self, x, cross, x_mask=None, cross_mask=None):
    '''
    x:[B,L,d_model]: generates query 
    cross:[B,L,d_model]:generates key and val 
    layers: list of decoder_layers
    '''
    #print('Decoder forward is called ...')
    for layer in self.layers:
      #print("Dec_layer is called inside decoder")
      x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

    if self.norm is not None:
      x = self.norm(x)

    if self.projection is not None:
      x = self.projection(x)

    return x

if __name__ == "__main__":
  my_att = FullAttention(output_attention=False)
  my_att_layer = AttentionLayer(my_att,16,4) # #d_model=16 ,num_head=4
  print(my_att_layer)

  x = torch.randn(2,10,16)
  crosss = torch.randn(2,10,16)

  #Test attention_layer:
  #qq = my_att_layer(x,x,x,None) #torch.Size([2, 10, 16])

  #Test encoder Layer:
  print("test_encoder_layer:")
  my_enc_layer =  EncoderLayer(my_att_layer,16)
  qqq = my_enc_layer(x)
  print(qqq[0].shape)
  print("============================================")
  print("test Encoder:")
  myEnc = Encoder([my_enc_layer,my_enc_layer,my_enc_layer])
  qqqq = myEnc(x)
  print(qqqq[0].shape)
  print("============================================")
  print("test Dec_layer:")
  myDec_layer = DecoderLayer(my_att_layer,my_att_layer,16)
  qqqqq = myDec_layer(x,crosss)
  print(qqqqq.shape)
  print("============================================")

  print("test Decoder:")
  my_Dec = Decoder([myDec_layer,myDec_layer,myDec_layer])
  tt = my_Dec(x,crosss)
  print(tt.shape)

