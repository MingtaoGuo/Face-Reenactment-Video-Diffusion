import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class InternalFeatureModulator(nn.Module):
    """Feature fusion block."""

    def __init__(self, lateral_channels, in_channels):
        """Init.

        Args:
            features (int): number of features
        """
        super(InternalFeatureModulator, self).__init__()

        self.resLateral = ResnetBlock2D(in_channels=lateral_channels, out_channels=lateral_channels, temb_channels=None)
        self.resStraight = ResnetBlock2D(in_channels=in_channels, out_channels=lateral_channels, temb_channels=None)

    def forward(self, lateral, x):
        """Forward pass.

        Returns:
            tensor: output
        """
        
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.resStraight(x, temb=None) 

        x += self.resLateral(lateral, temb=None)

        return x
