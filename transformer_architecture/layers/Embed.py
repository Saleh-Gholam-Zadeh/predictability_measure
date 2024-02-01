import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):  #d_model = n_embed ~64
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
        # print("pe.shape:",pe.shape) #pe.shape: torch.Size([1, 5000, 256])

    def forward(self, x):
        return self.pe[:, :x.size(1)]  #  #pe.shape: torch.Size([1, 75, 256])  #[1,ctx,d_model]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        #ORIGINAL KERNEL_SIZE=3
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model//8,
                                   kernel_size=1, padding=0, padding_mode='circular', bias=False) # d_model ta filter e  1*3 barmigardune
                                  #self.tokenConv.weight.shape : torch.Size([64, 1, 3])

        #originally it was only the below one
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False) # d_model ta filter e  1*3 barmigardune
        #                           #self.tokenConv.weight.shape : torch.Size([64, 1, 3])

        padding2 = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv_2 = nn.Conv1d(in_channels=c_in, out_channels=d_model//8,
                                     kernel_size=3, padding=padding2, padding_mode='circular', bias=False) # d_model ta filter e  1*3 barmigardune


        padding3 = 2 if torch.__version__ >= '1.5.0' else 4
        self.tokenConv_3 = nn.Conv1d(in_channels=c_in, out_channels=d_model//8,
                                     kernel_size=5, padding=padding3, padding_mode='circular', bias=False) # d_model ta filter e  1*5 barmigardune

        padding4 = 3 if torch.__version__ >= '1.5.0' else 6
        self.tokenConv_4 = nn.Conv1d(in_channels=c_in, out_channels=d_model//8,
                                     kernel_size=7, padding=padding4, padding_mode='circular', bias=False) # d_model ta filter e  1*5 barmigardune

        padding5 =  (c_in-1)//2 if torch.__version__ >= '1.5.0' else c_in-1
        self.tokenConv_5 = nn.Conv1d(in_channels=c_in, out_channels=d_model//2,
                                     kernel_size=c_in, padding=padding5, padding_mode='circular', bias=False) # d_model ta filter e  1*5 barmigardune


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    #original one
    # def forward(self, x):
    #     x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
    #
    #     return x
    # end of original one

    def forward(self, x):
        x1 = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2 )  #[350,75,d_model/4]

        #print("x1.shape:",x1.shape)
        x2 = self.tokenConv_2(x.permute(0, 2, 1)).transpose(1, 2) #[350,75,d_model/4]
        #print("x2.shape:", x2.shape)
        x3 = self.tokenConv_3(x.permute(0, 2, 1)).transpose(1, 2) #[350,75,d_model/4]
        #print("x3.shape:", x3.shape)
        x4 = self.tokenConv_4(x.permute(0, 2, 1)).transpose(1, 2) #[350,75,d_model/4]
        #print("x4.shape:", x4.shape)
        x5 = self.tokenConv_5(x.permute(0, 2, 1)).transpose(1, 2) #[350,75,d_model/4]
        #print("x5.shape:", x5.shape)
        x = torch.cat([x1,x2,x3,x4,x5] , dim=-1)
        #print("x.shape:",x.shape)
        return x



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
    #def forward(self, x):
      '''
      input: concat([x,x_mark__pred_len])

      '''
      if x_mark is not None:
        x = torch.cat([x,x_mark],dim=-2) #concat on time dim
      x = self.value_embedding(x)  + self.position_embedding(x) # + self.temporal_embedding(x_mark)
      return self.dropout(x)