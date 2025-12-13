import numbers
import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .models import register
from .module.mamba import ExtendedMamba
from .module.scan import ScanTransform
from .hybrid_block import HybridBlock
from .module.common import LayerNorm, to_3d, to_4d



## Input transform
# BLC -> BCHW
def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x

# BCHW -> BLC
def feature2token(x, norm=None):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    if norm:
        x = norm(x)
    return x
    
class FFN(nn.Module):
    def __init__(self, dim, bias, ffn_expansion_factor=2):
        super().__init__()

        ffn_channel = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        self.gelu = nn.GELU()
        self.project_out = nn.Conv2d(in_channels=ffn_channel, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.gelu(x)
        x = self.project_out(x)
        return x

# Custom Channel Attention Block
class CCABlock(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        
        # 1x1 conv + 3x3 dw conv
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=bias)
        
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            nn.Sigmoid()
        )

        # final 1x1 conv
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.ca(x)
        x = self.conv3(x)
        return x

## Simple Baselines for Image Restoration
# Note, default bias is set to True in paper, but we set it to False in our implementation.
# In : 2c, out : c
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimpleFFN(nn.Module):
    def __init__(self, dim, bias, ffn_expansion_factor=2):
        super(SimpleFFN, self).__init__()

        ffn_channel = int(dim * ffn_expansion_factor)
        assert ffn_channel % 2 == 0, 'FFN channel must be divisible by 2'

        self.project_in = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        self.simple_gate = SimpleGate()
        self.project_out = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.simple_gate(x)
        x = self.project_out(x)
        return x

## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MambaOnlyBlock(nn.Module):
    def __init__(self, dim, bias, layernorm_type, scan_transform, mamba_cfg=None, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = LayerNorm(dim, layernorm_type)
        self.mamba = ExtendedMamba(dim, scan_transform, **mamba_cfg)

    def forward(self, x):
        b, c, h, w = x.shape
        x_size = (h, w)

        def mamba_forward(t):
            y = self.mamba(feature2token(self.norm(t)), x_size)
            return token2feature(y, x_size)

        if self.use_checkpoint:
            return checkpoint(mamba_forward, x, use_reentrant=False)
        else:
            return mamba_forward(x)



# Metaformer-like mamba block
class MambaFormerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, layernorm_type, scan_transform,
                 mamba_cfg=None, 
                 use_checkpoint=False,
                 channel_mixer_type='GDFN'):
        super(MambaFormerBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm(dim, layernorm_type)
        self.mamba = ExtendedMamba(dim, scan_transform, **mamba_cfg)

        self.norm2 = LayerNorm(dim, layernorm_type)
        self.ffn = get_channel_mixer_layer(channel_mixer_type, dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b,c,h,w = x.shape
        x_size = (h,w)
        pre_x = x
        x = self.mamba(feature2token(self.norm1(x)), x_size)
        x = pre_x + token2feature(x, x_size)

        if self.use_checkpoint:
            x = x + checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        else:
            x = x + self.ffn(self.norm2(x))
        return x

# Custom FFN
ALLOWED_CHANNEL_MIXER_TYPE = ['GDFN', 'Simple', 'FFN', 'CCA']
ALLOWED_CHANNEL_MIXER_TYPE = [x.lower() if x is not None else x for x in ALLOWED_CHANNEL_MIXER_TYPE]
def get_channel_mixer_layer(type, dim, ffn_expansion_factor, bias):
    if type == 'ffn':
        return FFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    elif type == 'cca':
        return CCABlock(dim=dim, bias=bias)
    elif type == 'simple':
        return SimpleFFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    elif type == 'gdfn':
        return GDFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    else:
        raise NotImplementedError(f'Channel mixer type {type} is not implemented.')
    
def create_blocks(num_blocks, dim, ffn_expansion_factor, bias, layernorm_type, scan_transform,
                    mamba_cfg, checkpoint_percentage, channel_mixer_type='GDFN', *,
                    hybrid=False, heads=8, window_size=7):
    blocks = []
    num_checkpointed = math.ceil(checkpoint_percentage * num_blocks)
    for i in range(num_blocks):
        use_checkpoint = i < num_checkpointed

        if hybrid:
            mamba_layer = MambaOnlyBlock(
                dim=dim,
                bias=bias,
                layernorm_type=layernorm_type,
                scan_transform=scan_transform,
                mamba_cfg=mamba_cfg,
                use_checkpoint=use_checkpoint,
            )

            channel_mixer = get_channel_mixer_layer(channel_mixer_type, dim, ffn_expansion_factor, bias)

            block = HybridBlock(
                mamba_layer=mamba_layer,
                dim=dim,
                channel_mixer=channel_mixer,
                layernorm_type=layernorm_type,
                heads=heads,
                window_size=window_size,
                use_checkpoint=use_checkpoint,
            )

        else:
            block = MambaFormerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                    layernorm_type=layernorm_type, scan_transform=scan_transform, mamba_cfg=mamba_cfg, 
                                    use_checkpoint=use_checkpoint, channel_mixer_type=channel_mixer_type)
        blocks.append(block)
    return nn.Sequential(*blocks)

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Upscale(nn.Sequential):
    """Upscale module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upscale, self).__init__(*m)


##---------- EAmamba -----------------------
@register('eamamba')
class EAMamba(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=64,
        num_blocks=[4, 6, 6, 7], 
        num_refinement_blocks=2,
        ffn_expansion_factor=2.0,
        bias=False,
        layernorm_type='WithBias',   # other option 'BiasFree'
        dual_pixel_task=False,       # True for dual-pixel defocus deblurring only. Also set inp_channels=6
        checkpoint_percentage=0.0,   # percentage of checkpointed block
        channel_mixer_type='Simple',
        upscale=1,  
        mamba_cfg=None,

        use_hybrid=True,
        hybrid_stages=None,     # e.g. [False, False, True, True]
        attn_heads=8,
        window_size=7,
        


        **kwargs        # This is to ignore any other arguments that are not used
    ):

        super(EAMamba, self).__init__()
        if not use_hybrid:
            hybrid_stages = [False] * len(num_blocks)
        if hybrid_stages is None:
            hybrid_stages = [False] * len(num_blocks)

        assert len(hybrid_stages) == len(num_blocks), \
            f"hybrid_stages length {len(hybrid_stages)} must match num_blocks length {len(num_blocks)}"
        self.upscale = upscale
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        channel_mixer_type = channel_mixer_type.lower() if channel_mixer_type is not None else None
        assert channel_mixer_type in ALLOWED_CHANNEL_MIXER_TYPE, \
            f"channel_mixer_type should be one of {ALLOWED_CHANNEL_MIXER_TYPE}, but got {channel_mixer_type}"

        scan_type = mamba_cfg.get('scan_type')
        scan_type = scan_type.lower() if scan_type is not None else None
        scan_count = mamba_cfg.get('scan_count')
        scan_merge_method = mamba_cfg.get('scan_merge_method')

        # Custom scan input transform class
        scan_transform = ScanTransform(scan_type, scan_count, scan_merge_method)

        self.mamba_cfg = mamba_cfg
        shared_settings = { # settings that are the same for all stages
            'ffn_expansion_factor': ffn_expansion_factor,
            'bias': bias,
            'layernorm_type': layernorm_type,
            'scan_transform': scan_transform,
            'mamba_cfg': mamba_cfg,
            'checkpoint_percentage': checkpoint_percentage,
            'channel_mixer_type': channel_mixer_type
        }

        self.encoder_level1 = create_blocks(
            num_blocks=num_blocks[0], dim=dim, **shared_settings
        )
    
        self.down1_2 = Downsample(dim)
        
        self.encoder_level2 = create_blocks(
            num_blocks=num_blocks[1], dim=int(dim*2**1), **shared_settings
        )
        
        self.down2_3 = Downsample(int(dim*2**1))
        
        self.encoder_level3 = create_blocks(
            num_blocks=num_blocks[2], dim=int(dim*2**2), **shared_settings, hybrid=hybrid_stages[2],
            heads=attn_heads, window_size=window_size
        )

        self.down3_4 = Downsample(int(dim*2**2))
        
        self.latent = create_blocks(
            num_blocks=num_blocks[3], dim=int(dim*2**3), **shared_settings, hybrid=hybrid_stages[3],
            heads=attn_heads, window_size=window_size
        )
        
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = create_blocks(
            num_blocks=num_blocks[2], dim=int(dim*2**2), **shared_settings, hybrid=hybrid_stages[2],
            heads=attn_heads, window_size=window_size
        )

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        
        self.decoder_level2 = create_blocks(
            num_blocks=num_blocks[1], dim=int(dim*2**1), **shared_settings
        )
        
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = create_blocks(
            num_blocks=num_blocks[0], dim=int(dim*2**1), **shared_settings
        )
        
        self.refinement = create_blocks(
            num_blocks=num_refinement_blocks, dim=int(dim*2**1), **shared_settings
        )
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        # Add Upsample for the final output resolution
        if self.upscale > 1:
            self.upsample = Upscale(self.upscale, out_channels)

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        inp_enc_level1 = self.patch_embed(x)  
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            final_output = self.output(out_dec_level1)
        else:
            if self.upscale > 1:
                x_upscaled = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
                final_output = self.upsample(out_dec_level1) + x_upscaled
            else:
                final_output = self.output(out_dec_level1) + x

        return final_output

##---------- EAmamba SR -----------------------
@register('eamambasr')
class EAMambaSR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=64,
        num_blocks=[4, 4, 4, 4], 
        ffn_expansion_factor=2.0,
        bias=False,
        layernorm_type='WithBias',   # other option 'BiasFree'
        checkpoint_percentage=0.0,   # percentage of checkpointed block
        channel_mixer_type='Simple',
        upscale=2,                   # upscaling the final output
        mamba_cfg=None,
        **kwargs        # This is to ignore any other arguments that are not used
    ):
 
        super(EAMambaSR, self).__init__()
 
        self.upscale = upscale
 
        channel_mixer_type = channel_mixer_type.lower() if channel_mixer_type is not None else None
        assert channel_mixer_type in ALLOWED_CHANNEL_MIXER_TYPE, \
            f"channel_mixer_type should be one of {ALLOWED_CHANNEL_MIXER_TYPE}, but got {channel_mixer_type}"
 
        scan_type = mamba_cfg.get('scan_type')
        scan_type = scan_type.lower() if scan_type is not None else None
        scan_count = mamba_cfg.get('scan_count')
        scan_merge_method = mamba_cfg.get('scan_merge_method')
 
        # Custom scan input transform class
        scan_transform = ScanTransform(scan_type, scan_count, scan_merge_method)
 
        self.mamba_cfg = mamba_cfg
        shared_settings = { # settings that are the same for all stages
            'ffn_expansion_factor': ffn_expansion_factor,
            'bias': bias,
            'layernorm_type': layernorm_type,
            'scan_transform': scan_transform,
            'mamba_cfg': mamba_cfg,
            'checkpoint_percentage': checkpoint_percentage,
            'channel_mixer_type': channel_mixer_type
        }
 
        # 1. shallow feature extraction
        self.conv_first = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
 
        # 2. deep feature extraction
        self.num_layers = len(num_blocks)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(ResidualGroup(num_blocks[i], dim, shared_settings))
        self.norm = LayerNorm(dim, layernorm_type)
        self.conv_after_body = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
 
        # 3. upsampling
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upscale(upscale, dim)
        self.conv_last = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
 
 
    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
 
    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x
 
class ResidualGroup(nn.Module):
    def __init__(self, num_blocks, dim, shared_settings):
        super(ResidualGroup, self).__init__()
        self.blocks = create_blocks(num_blocks, dim, **shared_settings)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
 
    def forward(self, x):
        return self.conv(self.blocks(x)) + x
