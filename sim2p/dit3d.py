# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT-3D: https://github.com/DiT-3D/DiT-3D
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Clinical data                #
#################################################################################

class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=32, patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        voxel_size = (voxel_size, voxel_size, voxel_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TabEmbedder(nn.Module):
    def __init__(self, hidden_size, in_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )
    
    def forward(self, tab):
        tab_emb = self.mlp(tab)
        return tab_emb


#################################################################################
#                                 Core 3D DiT Model                             #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # from block_kwargs find tab_mode
        self.tab_mode = block_kwargs.get('tab_mode', 'concat')

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size if self.tab_mode != 'concat' else hidden_size * 2, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, **block_kwargs):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)

        # from block_kwargs find tab_mode
        self.tab_mode = block_kwargs.get('tab_mode', 'concat')

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size if self.tab_mode != 'concat' else hidden_size * 2, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a 3D Transformer backbone.
    """
    def __init__(
        self,
        image_size=80,
        patch_size=4,
        in_channels=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False,
        condition_mode='concat',
        tab_dim=26,
        tab_mode='concat',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels * 2 if condition_mode == 'concat' else in_channels
        self.condition_mode = condition_mode
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.tab_dim = tab_dim
        self.tab_mode = tab_mode

        print('DiT: image_size:', image_size, 'patch_size:', patch_size, 'hidden_size:', hidden_size, 'depth:', depth, 'num_heads:', num_heads, 'condition_mode:', condition_mode, 'tab_mode:', tab_mode)

        self.image_size = image_size

        self.x_embedder = PatchEmbed_Voxel(image_size, patch_size, self.in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        if tab_dim != 0:
            assert tab_dim > 0, "clinical data dimension must be a positive integer."
            self.tab_embedder = TabEmbedder(hidden_size, tab_dim)

        # fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        block_kwargs = {}
        block_kwargs['tab_mode'] = tab_mode
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, **block_kwargs)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.image_size//self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize clinical data embedding MLP:
        if self.tab_dim != 0:
            nn.init.normal_(self.tab_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.tab_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)
        voxels: (N, C, X, Y, Z)
        """
        c = self.out_channels
        p = self.patch_size
        x = y = z = self.image_size // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        imgs = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return imgs

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
    
    
    def forward(self, x, t, xT=None, **model_kwargs):
        """
        Forward pass of DiT.
        x: (N, C, X, Y, Z) tensor of 3D inputs 
        t: (N,) tensor of diffusion timesteps
        xT: (N, C, X, Y, Z) tensor of 3D condition inputs
        """
                
        if self.condition_mode == 'concat':
            x = torch.cat([x, xT], dim=1)
        
        x = self.x_embedder(x)

        x = x + self.pos_embed 

        t = self.t_embedder(t)  
        c = t

        if self.tab_dim != 0:
            tabular_data = model_kwargs.get('tabular', None)
            assert tabular_data is not None, "Clinical data should be provided."
            tab = self.tab_embedder(tabular_data)

            if self.tab_mode == 'concat':
                c = torch.cat([c, tab], dim=1)
            else:
                raise ValueError(f"Unsupported mode for clinical data integration: {self.tab_mode}")

        for block in self.blocks:
            x = block(x, c)                      
        x = self.final_layer(x, c)                
        x = self.unpatchify_voxels(x)

        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                 #
#################################################################################

def DiT_XL_2(pretrained=False, **kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(pretrained=False, **kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(pretrained=False, **kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_B_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

def DiT_B_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=32, num_heads=12, **kwargs)

def DiT_S_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_S_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)

def DiT_S_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=32, num_heads=6, **kwargs)

DiT3D_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}