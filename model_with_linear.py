from torch import nn
import copy
from collections import OrderedDict 
import torch
import torch.nn.functional as F 
from conformer.model import *
from conformer.layers import Swish
from conformer.attention import ScaledDotProductAttention
import numpy as np


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_blks):
        super(Encoder, self).__init__()
        self.conformers=ConformerEncoder(num_blocks=n_blks,
        d_model=d_model,
        num_heads=n_heads)

    def forward(self, x):
        return  self.conformers(x)


class Decoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_blks=8, reduction_factor=0.5, dropout=0.1, n_mel=80, n_linear=256, n_wav2vec2=768):
        super(Decoder, self).__init__()
        self.conformers=ConformerEncoder(num_blocks=n_blks,
        d_model=d_model,
        num_heads=n_heads)

        reduce_dim = int(d_model * reduction_factor)

        
        self.proj2mel = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, reduce_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(reduce_dim, n_mel),
        )
        
        self.proj2linear = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, reduce_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(reduce_dim, n_linear),
        )


        self.downsample=nn.Upsample(scale_factor=0.5)
       

    def forward(self, x):
        out=self.conformers(x)  # B x 100 x 512
        
        linear=self.proj2linear(out)
        mel=self.proj2mel(out)

        return  linear, mel 


class Speech_Enhancer(nn.Module):
    def __init__(self,  d_model_spk=256,  n_enc_heads=8, n_enc_attn=4, 
                     n_out_seq=100, n_dec_heads=8, dec_d_model=768, n_dec_attn=8, 
                    reduction_factor=0.5, dropout=0.1, output_mel=80, output_linear=256):

        super(Speech_Enhancer, self).__init__()
            
        
        self.encoder_spk=Encoder(d_model_spk, n_enc_heads, n_enc_attn)  # B x 50 x 256

        self.encoder_linear_spec=Encoder(d_model_spk, n_enc_heads, n_enc_attn)  # B x 100 x 257

        

        self.upsample_spk=nn.Upsample(n_out_seq)    # B x 512 x 100
        

        self.decoder=Decoder(dec_d_model, n_dec_heads, n_dec_attn, reduction_factor, dropout, output_mel, output_linear) # B x 100 x 80/257
        

    
    def forward(self, x_spk, noisy_linear_spec, noisy_mel):
        
        out_spk=self.encoder_spk(x_spk)
        out_spk=out_spk.permute(0, 2, 1)
        out_spk=self.upsample_spk(out_spk)
        out_spk=out_spk.permute(0, 2, 1)
        
        out_spec=self.encoder_linear_spec(noisy_linear_spec)
        

        enc_out=torch.cat([out_spec, out_spk],dim=2) 
        
        mask_lin, mask_mel=self.decoder(enc_out)
        
        
        return  (mask_lin + noisy_linear_spec), (noisy_mel + mask_mel)





