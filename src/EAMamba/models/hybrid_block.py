import torch
import torch.nn as nn
from .module.common import LayerNorm
from torch.utils.checkpoint import checkpoint

# --------------
# Attention Modules
# --------------

import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, dim, heads=8, window_size=7):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, heads)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad W then H
        Hp, Wp = x.shape[-2:]
        nH, nW = Hp // ws, Wp // ws

        # (B, C, nH, ws, nW, ws)
        x = x.view(B, C, nH, ws, nW, ws)
        # (B, nH, nW, ws, ws, C)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # windows as batch: (B*nH*nW, ws*ws, C)
        x = x.view(B * nH * nW, ws * ws, C)
        # to MHA format: (L, N, E) = (ws*ws, B*nH*nW, C)
        x = x.permute(1, 0, 2)

        x, _ = self.attn(x, x, x)

        # back: (B*nH*nW, ws*ws, C)
        x = x.permute(1, 0, 2).contiguous()
        # (B, nH, nW, ws, ws, C)
        x = x.view(B, nH, nW, ws, ws, C)
        # (B, C, nH, ws, nW, ws) -> (B, C, Hp, Wp)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)

        # crop back
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

class GlobalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.view(B, C, H*W).permute(2, 0, 1)
        x_, _ = self.attn(x_, x_, x_)
        x_ = x_.permute(1, 2, 0).view(B, C, H, W)
        return x_

# --------------
# Hybrid Block
# --------------

class HybridBlock(nn.Module):
    def __init__(self, mamba_layer, dim, channel_mixer, layernorm_type='WithBias',
                 heads=8, window_size=7, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.mamba = mamba_layer

        self.norm_local  = LayerNorm(dim, layernorm_type)
        self.norm_global = LayerNorm(dim, layernorm_type)
        self.norm_ffn    = LayerNorm(dim, layernorm_type)

        self.local_attn  = LocalAttention(dim, heads, window_size)
        self.global_attn = GlobalAttention(dim, heads)

        self.ffn = channel_mixer  # 直接用原本的 mixer (GDFN/Simple/FFN/CCA)

    def forward(self, x):
        # Mamba block already contains its own norm+ffn inside, but you wrapped it as a module.
        # Here we treat it as "token-mixer" stage with residual.
        res = x
        x = self.mamba(x)
        x = x + res

        # Local attn
        res = x
        x = self.local_attn(self.norm_local(x))
        x = x + res

        # Global attn
        res = x
        x = self.global_attn(self.norm_global(x))
        x = x + res

        # Channel mixer
        res = x
        if self.use_checkpoint:
            x = x + checkpoint(self.ffn, self.norm_ffn(x), use_reentrant=False)
        else:
            x = x + self.ffn(self.norm_ffn(x))
        return x
