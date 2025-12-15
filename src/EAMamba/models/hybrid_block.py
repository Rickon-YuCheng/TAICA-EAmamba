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

# class GlobalAttentionKVPool(nn.Module):
#     def __init__(self, dim, heads=8, kv_pool=4):
#         super().__init__()
#         self.dim = dim
#         self.heads = heads
#         self.kv_pool = kv_pool
#         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

#         # 用平均池化把 K/V token 數變少
#         self.pool = nn.AvgPool2d(kernel_size=kv_pool, stride=kv_pool)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape

#         # Q: full tokens
#         q = x.flatten(2).transpose(1, 2)   # (B, HW, C)

#         # K/V: pooled tokens
#         xp = self.pool(x)                  # (B, C, H', W')
#         k = xp.flatten(2).transpose(1, 2)  # (B, H'W', C)
#         v = k

#         out, _ = self.attn(q, k, v, need_weights=False)  # (B, HW, C)
#         out = out.transpose(1, 2).view(B, C, H, W)
#         return out


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
        # self.norm_global = LayerNorm(dim, layernorm_type)
        self.norm_ffn    = LayerNorm(dim, layernorm_type)

        self.local_attn  = LocalAttention(dim, heads, window_size)
        # self.global_attn = GlobalAttentionKVPool(dim, heads)

        self.ffn = channel_mixer  # 直接用原本的 mixer (GDFN/Simple/FFN/CCA)

    def forward(self, x):
        # ---- Mamba sublayer ----
        res = x
        if self.use_checkpoint:
            def mamba_forward(t):
                return self.mamba(t)
            x = res + checkpoint(mamba_forward, x, use_reentrant=False)
        else:
            x = res + self.mamba(x)

        # ---- Local attn sublayer ----
        res = x
        if self.use_checkpoint:
            def local_forward(t):
                return self.local_attn(self.norm_local(t))
            x = res + checkpoint(local_forward, x, use_reentrant=False)
        else:
            x = res + self.local_attn(self.norm_local(x))

        # # ---- Global attn sublayer ----
        # res = x
        # if self.use_checkpoint:
        #     def global_forward(t):
        #         return self.global_attn(self.norm_global(t))
        #     x = res + checkpoint(global_forward, x, use_reentrant=False)
        # else:
        #     x = res + self.global_attn(self.norm_global(x))

        # ---- FFN / channel mixer sublayer ----
        res = x
        if self.use_checkpoint:
            def ffn_forward(t):
                return self.ffn(self.norm_ffn(t))
            x = res + checkpoint(ffn_forward, x, use_reentrant=False)
        else:
            x = res + self.ffn(self.norm_ffn(x))

        return x
