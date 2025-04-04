import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 1
    n_labels: int = 16
    rms_norm: bool = True
    norm_eps: float = 1e-5
    max_position_embeddings: int = 20
    rope_theta: float = 10000
    mlp_bias: bool = True
    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

    
class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        bias = args.mlp_bias
        self.fc1 = nn.Linear(args.dim, args.dim, bias=bias) # Gate projection
        self.fc2 = nn.Linear(args.dim, args.dim, bias=bias) # Up projection
        self.fc3 = nn.Linear(args.dim, args.dim, bias=bias) # Down Projection
        self.act = nn.SiLU() 
        
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        x = self.fc3(self.act(self.fc1(x) * self.fc2(x)))
        return x
    
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads
        
        self.wq = nn.Linear(self.dim, self.dim, bias = False)
        self.wk = nn.Linear(self.dim, self.dim, bias = False)
        self.wv = nn.Linear(self.dim, self.dim, bias = False)
        if self.n_heads > 1:
            self.wo = nn.Linear(self.dim, self.dim, bias = False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=args.max_position_embeddings, base=args.rope_theta)
        self._init_weights()
        
    def _init_weights(self):
        if self.n_heads > 1:
            nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)
        for layer in [self.wq, self.wk, self.wv]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x:torch.Tensor, mask: torch.Tensor, output_attn_weights: bool = False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        queries = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        keys = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        values = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        # Apply RoPE
        cos, sin = self.rotary_emb(values, seq_len=seqlen)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        
        attn_weights = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(queries)
        output = torch.matmul(attn_weights, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        if self.n_heads > 1:
            output = self.wo(output)
        if output_attn_weights:
            return output, attn_weights
        else:
            return output

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.layer_id = layer_id
        self.args = args
        if args.rms_norm:
            self.attn_norm = RMSNorm(self.dim, args.norm_eps)
            self.mlp_norm = RMSNorm(self.dim, args.norm_eps)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.args.rms_norm:  
            hidden_states = x + self.attn(self.attn_norm(x), mask)
            out = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        else:
            hidden_states = x + self.attn(x, mask)
            out = hidden_states + self.mlp(hidden_states)
        return out
    

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        if args.rms_norm:   
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.out = nn.Linear(args.dim, args.n_labels, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        for layer in self.layers:
            x = layer(x, mask)
        if self.args.rms_norm:
            x = self.norm(x)
        # Take the last token and feed it to the output layer
        x = self.out(x[:, -1, :])
        return x

