# simple transformer architecture using torch modules

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding
import math

class FFN(nn.Module):
    def __init__(self, d_model=512, inner_scale=4, dtype=torch.bfloat16):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model*inner_scale, dtype=dtype)
        self.fc2 = nn.Linear(d_model*inner_scale, d_model, dtype=dtype)
        self.f = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.f(x)
        x = self.fc2(x)
        return x 

class Encoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dtype=torch.bfloat16):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dtype=dtype, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.ffn = FFN(d_model=embed_dim)

    def forward(self, x):
        atn, _ = self.mha(x, x, x)
        out = self.norm1(x + atn)
        ffn = self.ffn(out)
        out = self.norm2(out + ffn)
        return out

class Decoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.masked_mha = nn.MultiheadAttention(embed_dim, num_heads, dtype=dtype, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dtype=dtype, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.norm3 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.ffn = FFN(d_model=embed_dim)

    def forward(self, x, e_in):
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1), dtype=self.dtype) # x is expected to be 3D (bsz, seq_len, d_model)
        m_atn, _ = self.masked_mha(x, x, x, attn_mask=mask)
        out = self.norm1(m_atn + x)
        atn, _ = self.mha(e_in, e_in, out)
        out1 = self.norm2(out + atn)
        ffn = self.ffn(out1)
        out2 = self.norm3(out1 + ffn)
        return out2


class PositionalEncoding(nn.Module):
    """pulled from https://machinelearningmastery.com/positional-encodings-in-transformer-models/"""
    def __init__(self, d_model, N=10_000, dtype=torch.bfloat16): # N should be larger than the maximum sequence length
        super().__init__()
        self.d_model = d_model
        _i = torch.arange(0, d_model//2, dtype=dtype)
        self.div = torch.exp(-np.log(N)*(2*_i/d_model))

    def forward(self, x):
        # remember that this function expects embedded inputs of shape seq_len, d_model
        seq_len = x.size(0)
        position = torch.arange(seq_len).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * self.div)
        pe[:, 1::2] = torch.cos(position * self.div)
        out = x + pe
        return out

class Transformer(nn.Module):
    def __init__(self, pos_encoding, input_embedding, output_embedding, num_layers=6):
        pass

if __name__ == "__main__":
    # x = torch.randn(32, 512, dtype=torch.bfloat16) # this is the shape post embedding layer(should also include bsz=1)
    input = torch.LongTensor([x for x in range(32)]) 
    pos_encoding = PositionalEncoding(d_model=512)
    embedding = Embedding(num_embeddings=32_000,embedding_dim=512)
    x = embedding(input)
    output = pos_encoding(x)
    print(output.shape) # currently supports only bsz = 1

    # x = torch.randn(10, 32, 512, dtype=torch.bfloat16) # single batch
    # xi = torch.rand(10, 32, 512, dtype=torch.bfloat16)

    # enc = Encoder()
    # dec = Decoder()

    # e_out = enc(x)
    # d_out = dec(xi, e_out)
    # print(d_out.shape)
