import torch.nn as nn
import torch.nn.functional as F
import torch
from transformer_architecture.layers.Embed import DataEmbedding
from .SelfAttention_Family import DSAttention, AttentionLayer

# class TriangularCausalMask():
#     def __init__(self, B, L, device="cpu"):
#         mask_shape = [B, 1, L, L]
#         with torch.no_grad():
#             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
#
#     @property
#     def mask(self):
#         return self._mask

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        print("original x.shape:", x.shape)
        x = self.downConv(x.permute(0, 2, 1))  # [B,E,S] its permuted to be applied on the time_dim not feature_dim
        print("downconv is called")
        print("after downConv x.shape:", x.shape)
        x = self.norm(x)
        print("after norm x.shape:", x.shape)
        x = self.activation(x)
        print("x.shape:", x.shape)
        x = self.maxpool(x)
        print("x.shape:", x.shape)
        x = x.transpose(1, 2)  ## why ??  probably to have features at the end again
        print("x.shape:", x.shape)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        #print("Encoder layer is created")
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''
    input:  x:[B,L,d_model]
    output: the same as x:[B,L,d_model]
    '''
        #print("Encoder_layer forward() is called ")
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, enc_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x:[B,L,D]
        attns = []
        if self.conv_layers is not None:

            # The reason why we only import delta for the first attn_block of Encoder
            # is to integrate Informer into our framework, where row size of attention of Informer is changing each layer
            # and inconsistent to the sequence length of the initial input,
            # then no way to add delta to every row, so we make delta=0.0 (See our Appendix E.2)
            #
            for i, (enc_layer, conv_layer) in enumerate(zip(self.enc_layers, self.conv_layers)):
                # print("conv layer is not None")
                # print("applying layers in Encoder")
                delta = delta if i == 0 else None
                x, attn = enc_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.enc_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for enc_layer in self.enc_layers:
                #print("applying layers in Encoder")
                x, attn = enc_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
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
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Note that delta only used for Self-Attention(x_enc with x_enc)
        # and Cross-Attention(x_enc with x_dec),
        # but not suitable for Self-Attention(x_dec with x_dec)
        '''
    x:[B,L,d_model] query from decoder #attention layer  was already passed to the __init__
    cross: [B,L,d_model] key and val from encoder
    '''
        #print('Decoder layer: attn_mask=', x_mask)

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta)[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        '''
    layers is a list of decoder layers
    '''
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        '''
    layers is a list of decoder layers
    '''
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


if __name__ == '__main__':

    from SelfAttention_Family import DSArrention
    # saleh_conv = ConvLayer(c_in=128)
    # print(saleh_conv)
    # x=torch.randn(2,72,128)
    # saleh_conv(x)
    #

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
    # output_1 , _ = saleh_att_layer(cur_x,cur_x,cur_x ,attn_mask=saleh_mask)
    # print("output_1.shape:",output_1.shape)

    saleh_enc_layer = EncoderLayer(attention=saleh_att_layer, d_model=d)
    # print(saleh_enc_layer)
    output_2, _ = saleh_enc_layer(cur_x)
    print("output_2.shape:", output_2.shape)

    ##only tested until encoder_layer

