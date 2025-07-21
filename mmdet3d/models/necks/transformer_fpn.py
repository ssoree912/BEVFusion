import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from torch.nn import TransformerEncoderLayer
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS

@NECKS.register_module()
class TransformerFPN(BaseModule):
    """Conv → TransformerEncoder → Conv 으로 이루어진 간단한 FPN."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 transformer_cfg,
                 strides=None,
                 upsample_strides=None,
                 init_cfg=None,
                 num_proposals=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_proposals = num_proposals
        # Ignore any extra config keys (e.g., norm_cfg, conv_cfg)
        _ = kwargs
        # Handle deprecated 'upsample_strides' key
        if upsample_strides is not None:
            strides = upsample_strides
        self.strides = strides

        embed_dim = transformer_cfg['embed_dims']
        # 1) lateral conv to embed_dim
        self.lateral_convs = nn.ModuleList([
            ConvModule(
                in_channels[i], embed_dim, kernel_size=1,
                conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
            ) for i in range(len(in_channels))
        ])
        # 2) N개의 TransformerEncoderLayer
        self.transformers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_cfg['num_heads'],
                dim_feedforward=transformer_cfg['feedforward_channels']
            )
            for _ in range(transformer_cfg['num_layers'])
        ])
        # 3) output conv back to out_channels
        self.output_convs = nn.ModuleList([
            ConvModule(
                embed_dim, out_channels, kernel_size=1,
                conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=None
            ) for _ in range(len(in_channels))
        ])
        self.num_outs = num_outs

    def forward(self, inputs):
        # inputs: list of feature maps from backbone (e.g. [P3, P4])
        outs = []
        for i, x in enumerate(inputs):
            B, C, H, W = x.shape
            x = self.lateral_convs[i](x)  # → (B, E, H, W)
            # flatten for Transformer: (H*W, B, E)
            x_flat = x.flatten(2).permute(2, 0, 1)
            for layer in self.transformers:
                x_flat = layer(x_flat)
            # back to (B, E, H, W)
            x2 = x_flat.permute(1, 2, 0).view(B, -1, H, W)
            out = self.output_convs[i](x2)  # → (B, out_channels, H, W)
            outs.append(out)
        # if num_outs > len(inputs), pad with last
        # align all feature maps to the resolution of the first level
        base_h, base_w = outs[0].shape[2], outs[0].shape[3]
        outs = [
            F.interpolate(o, size=(base_h, base_w), mode='bilinear', align_corners=False)
            for o in outs
        ]
        if self.num_outs > len(outs):
            for _ in range(self.num_outs - len(outs)):
                outs.append(outs[-1])
        return outs