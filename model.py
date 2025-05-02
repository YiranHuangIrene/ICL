import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    m1_dim: int = 128
    m2_dim: int = 128
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 1
    n_labels: int = 16
    rms_norm: bool = True
    rope: bool = True
    norm_eps: float = 1e-5
    max_position_embeddings: int = 20
    rope_theta: float = 10000
    mlp_bias: bool = True
    L_pos: int = 64

@dataclass
class MLPEncoderArgs:
    input_dim: int = 512
    hidden_sizes: List[int] = field(default_factory=lambda: [64])
    output_dim: int = 32
    num_classes: int = 256

@dataclass
class TransformerEncoderArgs:
    feat_dim: int = 512
    input_dim: int = 32
    output_dim: int = 32
    num_classes: int = 256
    num_layers: int = 1
    num_heads: int = 1

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
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads
        
        self.wq = nn.Linear(self.dim, self.dim, bias = False)
        self.wk = nn.Linear(self.dim, self.dim, bias = False)
        self.wv = nn.Linear(self.dim, self.dim, bias = False)
        if self.n_heads > 1:
            self.wo = nn.Linear(self.dim, self.dim, bias = False)
        if args.rope:
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
        if self.args.rope:
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
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, output_attn_weights: bool = False):
        if output_attn_weights:
            if self.args.rms_norm:  
                hidden_states , attn_weights = self.attn(self.attn_norm(x), mask, output_attn_weights)
                hidden_states = x + hidden_states 
                out = hidden_states + self.mlp(self.mlp_norm(hidden_states))
            else:
                hidden_states,  attn_weights = self.attn(x, mask, output_attn_weights)
                hidden_states = x + hidden_states 
                out = hidden_states  + self.mlp(hidden_states)
            return out, attn_weights
        else:
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
        
    def forward(self, x: torch.Tensor, output_attn_weights: bool = False):
        bsz, seqlen, _ = x.shape
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        if output_attn_weights:
            attn_weights = []
        for layer in self.layers:
            if output_attn_weights:
                x, attn_weight = layer(x, mask, output_attn_weights)
                attn_weights.append(attn_weight)
            else:
                x = layer(x, mask)
        if self.args.rms_norm:
            x = self.norm(x)
        # Take the last token and feed it to the output layer
        x = self.out(x[:, -1, :])
        if output_attn_weights:
            return x, attn_weights
        else:
            return x

class Projector(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.m2_dim, args.m1_dim, bias=False) 
        self.act = nn.GELU()
        self.fc2 = nn.Linear(args.m1_dim, args.m1_dim, bias=False) 
        self._init_weights()
        
    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            
    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)))
        return x
    
class MMTransformer(Transformer):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.projector = Projector(args)
    
    def combine_mm_input_seqs_v1(self, x_m1: torch.Tensor, x_m2: torch.Tensor):
        """
        x_m1: (S, 3N+1, D1)
        x_m2: Projected m2 feature, which has the same dimension as D1, shape (S, N+1, D1)
        """
        bsz = x_m1.shape[0]
        seq_len = x_m1.shape[1]
        feat_dim = x_m1.shape[2]
        x_m1[:,1:-1:3,:] = x_m2[:,:-1,:]
        x_m1[:,-1,:] = x_m2[:,-1,:]
        if self.args.rope:
            inputs = x_m1
        else:
            inputs = torch.zeros((bsz, seq_len, self.args.L_pos + feat_dim), device=x_m1.device, dtype=x_m1.dtype)
            inputs[:,:,self.args.L_pos:] = x_m1
            shifts = torch.randint(0, self.args.L_pos - seq_len + 1, size = (bsz,), device=x_m1.device)
            for s in range(bsz):
                inputs[s,:,shifts[s]:shifts[s] + seq_len] = torch.eye(seq_len, device=x_m1.device)
        return inputs
    
    def forward(self, x_m1: torch.Tensor, x_m2: torch.Tensor, output_attn_weights: bool = False):
        x_m2 = self.projector(x_m2)
        inputs = self.combine_mm_input_seqs_v1(x_m1, x_m2)
        return super().forward(inputs, output_attn_weights)
    
class MLPEncoder(nn.Module):
    def __init__(self, args, activation=nn.ReLU):
        """
        hidden_sizes: list of int for the first (n-1) hidden layers
        bottleneck_dim: d2, size of final hidden layer before classification head
        """
        super().__init__()
        self.input_dim = args.input_dim
        self.hidden_sizes = args.hidden_sizes
        self.bottleneck_dim = args.output_dim
        self.num_classes = args.num_classes
        
        layers = []
        prev_dim = self.input_dim
        # build all but final hidden
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        # bottleneck layer
        layers.append(nn.Linear(prev_dim, self.bottleneck_dim))
        layers.append(activation())
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.bottleneck_dim, self.num_classes)

    def forward(self, x, return_features=False):
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits

    def extract_features(self, x):
        """Get the d2-dimensional representation."""
        return self.feature_extractor(x)

class TransformerEncoder(nn.Module):
    def __init__(self, args, dropout=0.1):
        super().__init__()
        self.seq_len = args.feat_dim // args.input_dim
        # linear projection of each segment
        self.patch_embed = nn.Linear(args.input_dim, args.output_dim)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.output_dim))
        # positional embeddings for CLS + sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + 1, args.output_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.output_dim,
                                                   nhead=args.num_heads,
                                                   dim_feedforward=5 * args.output_dim,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
        # classification head
        self.classifier = nn.Linear(args.output_dim, args.num_classes)

    def forward(self, x):
        features = self.extract_features(x)  # (B, output_dim)
        logits = self.classifier(features)              # (B, num_classes)
        return logits
    
    def extract_features(self, x):
        # x: (batch, D)
        B, D = x.shape
        # reshape to (batch, seq_len, input_dim)
        x = x.view(B, self.seq_len, -1)
        # embed segments
        x = self.patch_embed(x)  # (B, seq_len, output_dim)
        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,output_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, seq_len+1, output_dim)
        x = x + self.pos_embedding                      # add positional embeddings
        # Transformer encoding
        x = self.encoder(x)                             # (B, seq_len+1, output_dim)
        # take CLS output as feature
        features = x[:, 0, :]                           # (B, output_dim)
        return features
    
class MLLMTransformer(Transformer):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.projector = Projector(args)
    
    def init_encoder(self, encoder):
        self.encoder = encoder
           
    def combine_mm_input_seqs_v1(self, x_m1: torch.Tensor, x_m2: torch.Tensor):
        """
        x_m1: (S, 3N+1, D1)
        x_m2: Projected m2 feature, which has the same dimension as D1, shape (S, N+1, D1)
        """
        bsz = x_m1.shape[0]
        seq_len = x_m1.shape[1]
        feat_dim = x_m1.shape[2]
        x_m1[:,1:-1:3,:] = x_m2[:,:-1,:]
        x_m1[:,-1,:] = x_m2[:,-1,:]
        if self.args.rope:
            inputs = x_m1
        else:
            inputs = torch.zeros((bsz, seq_len, self.args.L_pos + feat_dim), device=x_m1.device, dtype=x_m1.dtype)
            inputs[:,:,self.args.L_pos:] = x_m1
            shifts = torch.randint(0, self.args.L_pos - seq_len + 1, size = (bsz,), device=x_m1.device)
            for s in range(bsz):
                inputs[s,:,shifts[s]:shifts[s] + seq_len] = torch.eye(seq_len, device=x_m1.device)
        return inputs
    
    def forward(self, x_m1: torch.Tensor, x_m2: torch.Tensor, output_attn_weights: bool = False):
        x_m2 = self.encoder.extract_features(x_m2)
        x_m2 = self.projector(x_m2)
        inputs = self.combine_mm_input_seqs_v1(x_m1, x_m2)
        return super().forward(inputs, output_attn_weights)