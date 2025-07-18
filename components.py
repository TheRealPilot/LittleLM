# Smaller atomic units of transformer architecture. inefficient but educational

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.functional import scaled_dot_product_attention
from torch.nn import TransformerEncoderLayer, TransformerEncoder

# make an encoder

class Encoder(nn.Module):
    def __init__(self, stack: int=6, d_model: int=512, device=None)->None:
        super().__init__()
        self.stack = stack
        self.mha = nn.ModuleList([MHA(device=device) for i in range(stack)])
        # Also try with power normalization
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-08, device=device, dtype=torch.bfloat16) for i in range(stack)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-08, device=device, dtype=torch.bfloat16) for i in range(stack)])
        self.fc1 = nn.ModuleList([nn.Linear(d_model, d_model*4, device=device, dtype=torch.bfloat16) for i in range(stack)])
        self.fc2 = nn.ModuleList([nn.Linear(d_model*4, d_model, device=device, dtype=torch.bfloat16) for i in range(stack)])

    def forward(self, x: torch.Tensor):
        """Receives input from input embedding of size d_model"""
        for i in range(self.stack):
            x = self.norm1[i](x + self.mha[i](x))
            x = self.norm2[i](x + self.fc2[i](self.fc1[i](x)))
        return x            

class MHA(nn.Module):
    """Multihead attention from Attention is All You Need"""
    def __init__(self, num_heads: int=8, device=None)->None:
        super().__init__()
        self.d_model = num_heads * 64 # d_model = 512 here
        self.dk = self.dv = int(self.d_model / num_heads)
        # NOTE: one linear layer can be used instead of independent linear layers
        # self.wq_layers = nn.ModuleList([nn.Linear(self.d_model, self.dk, device=device, dtype=torch.bfloat16) for i in range(num_heads)])
        # self.wk_layers = nn.ModuleList([nn.Linear(self.d_model, self.dk, device=device, dtype=torch.bfloat16) for i in range(num_heads)])
        # self.wv_layers = nn.ModuleList([nn.Linear(self.d_model, self.dv, device=device, dtype=torch.bfloat16) for i in range(num_heads)])
        self.wo = nn.Linear(int(num_heads*self.dv), self.d_model, device=device, dtype=torch.bfloat16)
        self.heads = nn.ModuleList([ScaledDPattn() for i in range(num_heads)])

    def forward(self, x):
        stk = []
        for b in range(x.shape[0]):
            t = []

            # TODO: figure out why this is inequivalent to torch

            # for i, h in enumerate(self.heads):
            #     q = self.wq_layers[i](x[b, :, :])
            #     k = self.wk_layers[i](x[b, :, :])
            #     v = self.wv_layers[i](x[b, :, :])
            #     t.append(self.heads[i](q, k, v))
            out = torch.cat(t, dim=1)
            res = self.wo(out)
            stk.append(res)
        x = torch.stack(stk)
        return x

class ScaledDPattn(nn.Module):
    """Scaled dot product attention from Attention is All You Need"""
    def __init__(self)->None:
        super().__init__()

    def forward(self, q, k, v):
        assert q.shape[-1] == k.shape[-1], "q and k matrices not of same dimension (check last axis)"
        scale_factor = 1/math.sqrt(q.shape[-1])
        x = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, v)
        return x
    
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # ensure tensors are of axis=-1 shape d_model
    d_model = 512   
    torch.seed()
    x = torch.randn([10, 32, d_model], dtype=torch.bfloat16).to(device) # first dimension is batch size

    enc = Encoder(stack=1, device=device)
    layer = TransformerEncoderLayer(d_model=512, nhead=8, device=device, dtype=torch.bfloat16, batch_first=True) # this expects a Tensor of shape bsz, input_dimension, d_model
    torch_enc = TransformerEncoder(layer, num_layers=1)

    # print(f"*** custom encoder architecture ***")
    # for child in enc.children():
    #     print(child)
    # print()

    # print(f"*** torch encoder architecture ***")
    # for child in torch_enc.children():
    #     print(child)

    [p.data.fill_(0.01) for p in list(enc.parameters()) + list(layer.parameters())]
    result = enc(x)
    torch_result = torch_enc(x)

    print(f"encoder shape result: {result.shape}")
    print(f"torch encoder shape result: {torch_result.shape}")

    print(torch.allclose(result, torch_result, atol=1e-05)) # fails -> maybe due to different normalization layers?
    
    
    #atn = MHA(device=device)
    #torch_atn = torch.nn.MultiheadAttention(d_model, num_heads=8, device=device, dtype=torch.bfloat16)
    # set all parameters to the same value
    # [p.data.fill_(0.01) for p in list(atn.parameters()) + list(torch_atn.parameters())]

    # result = atn(x)
    # torch_result, weights = torch_atn(x, x, x)
    # print(f"custom MHA shape: {result.shape}")
    # print(f"torch MHA shape: {torch_result.shape}")
    
    # print(torch.allclose(result, torch_result, atol=1e-05)) # passes


