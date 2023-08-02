from ipaddress import ip_address
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from agg_block.pos_encoding import build_position_encoding

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

    
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
    
    
class PostNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = 'geglu', more_dropout = False, xavier_init = False):
        super().__init__()
        act_in_dim = int(dim * mult)
        act_out_dim = act_in_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError("Invalid activation function")
            
        self.net = nn.Sequential(
            nn.Linear(dim, act_in_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(act_out_dim, dim),
            nn.Dropout(dropout) if more_dropout else nn.Identity()
        )
        
        if xavier_init:
            self._reset_parameter()
    
    def _reset_parameter(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.net.apply(fn)

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, 
        query_dim, 
        context_dim = None, 
        heads = 8, dim_head = 64, 
        dropout = 0., 
        more_dropout = False, 
        xavier_init = False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.attn_holder = nn.Identity()
        
        self.attn_matrix_dropout = nn.Dropout(dropout) if more_dropout else nn.Identity()
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        if xavier_init:
            self._reset_parameter()
        
    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

    def forward(self, x, context = None, mask = None, k_pos = None, q_pos = None):
        h = self.heads

        q = self.to_q(x if q_pos is None else x + q_pos)
        context = default(context, x)
        k = self.to_k(context if k_pos is None else context + k_pos)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k)
        sim = sim * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = 1)
        attn = self.attn_holder(attn)
        attn = attn / (attn.sum(dim = -1, keepdim = True) + 1e-7)
        
        if torch.isnan(attn).any():
            import pdb; pdb.set_trace()
            
        attn = self.attn_matrix_dropout(attn)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)