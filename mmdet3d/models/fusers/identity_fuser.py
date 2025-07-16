from mmcv.runner import BaseModule
from mmdet3d.models.builder import FUSERS  

@FUSERS.register_module()
class IdentityFuser(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels[0] == out_channels, \
            f"In IdentityFuser, in_channels[0] ({in_channels[0]}) must match out_channels ({out_channels})"
        self.out_channels = out_channels

    def forward(self, features):
        # features: list with single feature tensor from camera
        return features[0]  # just return camera feature