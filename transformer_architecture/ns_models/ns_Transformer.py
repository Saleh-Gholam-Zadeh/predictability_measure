import torch
import torch.nn as nn
from transformer_architecture.ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from transformer_architecture.ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
from transformer_architecture.layers.Embed import DataEmbedding
from dataclasses import dataclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@dataclass
class NS_TSConfig:
    # block_size: int = 128   #0.0614 seconds context
    # vocab_size: int = 25   # in TS it is input dimension e.g 25 features // token embedding layer was changed to simple linear layer
    # n_layer: int = 6
    # n_head: int = 6
    # n_embed: int = 384 #C #every head is 768/12= 64 dimensional  we assumed also head_size=C= n_embed
    # head_size = n_embed//n_head #C = 64
    # dropout: float = 0.1
    # -----------------------------------
    enc_in: int = 1  # number of channels(features) in the input #would be replaced by number of features in the input
    dec_in: int = 1  # number of input channels(features) in the decoder #would be replaced by number of features in the input
    c_out: int = 1  # output dim. seems to be always the same as dec_in except for the classification schema
    # d_model: int = 512 #n_embed originally=512
    d_model: int = 512

    # dropout: float = 0.1
    dropout: float = 0.1

    pred_len: int = 1
    seq_len: int = 49  # is it equal to the context size S?
    label_len: int = seq_len//2 # from where it takes the encoder output. it can be S//2 (ctx//2)
    p_hidden_layers: int = 2
    factor: int = 3
    #p_hidden_dims = [128, 128]
    p_hidden_dims = [64, 64]

    train_res: bool = False



    output_attention = True  # whether to output attention in encoder

    n_heads: int = 8
    activation: str = 'gelu'
    e_layers: int = 2
    d_layers: int = 1

    embed = None
    freq = None
    d_ff = None  # here none will be translate to 4*d_model
    factor = None
    # n_embed: int = 384/// hamun d_model ast #C #every head is 768/12= 64 dimensional  we assumed also head_size=C= n_embed
    # head_size = n_embed//n_head #C = 64

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O # I checked: it returns B*1  for tau and B*S for delta

        return y


class Model(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.train_res = configs.train_res

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
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
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
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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

        print(" line 135 ns_transformer.py     configs.p_hidden_dims :",configs.p_hidden_dims)
        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()
        B,L,in_features = x_raw.size()

        # S --> sequence length
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-10).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros((B,self.pred_len,in_features)).to(x_enc.device) ], dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # print("enc_out.device:",enc_out.device)
        # print("enc_out.device:", enc_out.device)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta) #[B, pred_len+S//2 , out_feature]

        # De-normalization
        if not (self.train_res):
            #print("mean_added")
            dec_out = dec_out * std_enc + mean_enc
        else:
            #
            #print("mean NOT added")
            dec_out = dec_out * std_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]



if __name__ == "__main__":
  inp_enc = torch.randn(10,50,7).to(device)
  my_conf = NS_TSConfig(enc_in=inp_enc.shape[-1], dec_in=inp_enc.shape[-1] , seq_len=50 ,label_len=0 ,factor=3 , p_hidden_layers=2)
  print(my_conf)
  ts_model = Model(my_conf).to(device)
  #print(ts_model)
  inp_enc = torch.randn(10,50,7).to(device)
  inp_dec = torch.randn(10,10,7).to(device)
  mark_dec = torch.zeros((10,30,1)).to(device)
  a,b = ts_model(inp_enc,None,inp_dec,None)
  print(a.shape)
  #print("Attention weights:",b[0].shape , b[1].shape)