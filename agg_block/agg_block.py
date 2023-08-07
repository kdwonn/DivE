from math import pi, log
from functools import wraps

import torch
from torch import dropout, nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from agg_block.pos_encoding import build_position_encoding
from agg_block.attention import *
    

class AggregationBlock(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        pos_enc_type = 'none',
        pre_norm = True,
        post_norm = True, 
        activation = 'geglu',
        last_ln = False,
        ff_mult = 4,
        more_dropout = False,
        xavier_init = False,
        query_fixed = False,
        query_xavier_init = False,
        query_type = 'learned',
        encoder_isab = False,
        first_order=False
    ):
        """
        Args:
            depth: Depth of net.
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of element slots
            latent_dim: slot dimension.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.num_latents = num_latents
        self.query_type = query_type
        self.latent_dim = latent_dim
        self.encoder_isab = encoder_isab
        self.first_order = first_order
        
        if self.query_type == 'learned':
            self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim))
            if query_fixed:
                self.latents.requires_grad = False
            if query_xavier_init:
                nn.init.xavier_normal_(self.latents)
        elif self.query_type == 'slot':
            self.slots_mu = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
            self.slots_log_sigma = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
        else:
            raise NotImplementedError
        
        assert (pre_norm or post_norm)
        self.prenorm = PreNorm if pre_norm else lambda dim, fn, context_dim=None: fn
        self.postnorm = PostNorm if post_norm else nn.Identity
        ff = FeedForward
        
        # * decoder cross attention layers
        get_cross_attn = \
            lambda: self.prenorm(
                latent_dim, 
                Attention(
                    latent_dim, input_dim,
                    heads = 4, dim_head = 512, dropout = attn_dropout, more_dropout = more_dropout, xavier_init = xavier_init
                ), 
                context_dim = input_dim)
        
        get_cross_ff = lambda: self.prenorm(latent_dim, ff(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_cross_postnorm = lambda: self.postnorm(latent_dim)
        
        get_cross_attn, get_cross_ff = map(cache_fn, (get_cross_attn, get_cross_ff))

        self.layers = nn.ModuleList([])
        
        for i in range(depth):
            should_cache = i >= 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_postnorm(),
                get_cross_ff(**cache_args),
                get_cross_postnorm()
            ]))

        # Last FC layer
        assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not post_norm else nn.Identity()
        )
        
        self.encoder_output_holder = nn.Identity()
        self.decoder_output_holder = nn.Identity()
        
    def get_queries(self, b):
        if self.query_type == 'learned':
            ret = repeat(self.latents, 'n d -> b n d', b = b)
        elif self.query_type == 'slot':
            slots_init = torch.randn((b, self.num_latents, self.latent_dim)).cuda()
            ret = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        return ret

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data)
        
        data = rearrange(data, 'b ... d -> b (...) d')
        
        x = self.get_queries(b).type_as(data)

        for i, (cross_attn, pn1, cross_ff, pn2) in enumerate(self.layers):
            x = cross_attn(x, context = data, mask = mask, k_pos = pos, q_pos = None) + x
            x = pn1(x)
            x = cross_ff(x) + x
            x = pn2(x)
            
        x = self.decoder_output_holder(x)
        return self.last_layer(x)
