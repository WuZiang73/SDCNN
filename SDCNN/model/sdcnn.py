import torch
import torch.nn as nn
import model.ops as ops
import torch.nn.functional as F
from model.ops import MeanShift, UpsampleBlock
from model.DSConv import DSConv_pro

class SwishReLU(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, x, x / (1 + torch.exp(-x)))

class SDCNN(nn.Module):
    def __init__(self, **kwargs):
        super(SDCNN, self).__init__()

        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        features = 64
        channels = 3

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )

        # Feature extraction with DSConv_pro and residual connections
        self.conv2 = DSConv_pro(in_channels=features, out_channels=features, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True)
        self.conv3 = DSConv_pro(in_channels=features, out_channels=features, kernel_size=9, extend_scope=1.0, morph=1, if_offset=True)
        self.conv4 = DSConv_pro(in_channels=features, out_channels=features, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True)

        # Additional convolution layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features // 4, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv11_1x5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features // 4, kernel_size=(1, 5), padding=(0, 2), bias=False),
            SwishReLU()
        )
        self.conv11_5x1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features // 4, kernel_size=(5, 1), padding=(2, 0), bias=False),
            SwishReLU()
        )
        self.conv11_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features // 4, kernel_size=5, padding=2, bias=False),
            SwishReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv19 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv20 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            SwishReLU()
        )

        # Upsampling
        self.upsample = UpsampleBlock(64, scale=scale, multi_scale=multi_scale, group=1)
        self.conv17 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=3, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x, scale):
        x0 = self.sub_mean(x)
        x1 = self.conv1(x0)

        # Feature extraction with DSConv_pro and residual connections
        x2 = self.conv2(x1) + x1  # Residual connection
        x3 = self.conv3(x2) + x2  # Residual connection
        x4 = self.conv4(x3) + x3  # Residual connection

        # Combine features with residual connections
        x4_1 = x2 + x4  # Residual connection

        x5 = self.conv5(x4_1)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv8(x8)
        x10 = self.conv8(x9)
        x11 = self.conv11(x10)
        x11_1x5 = self.conv11_1x5(x10)
        x11_5x1 = self.conv11_5x1(x10)
        x11_5x5 = self.conv11_5x5(x10)
        x11_concat = torch.cat([x11, x11_1x5, x11_5x1, x11_5x5], dim=1)
        x12 = self.conv12(x11_concat)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)

        # Additional convolution layers with residual connections
        x18 = self.conv18(x15)  # Residual connection
        x19 = self.conv19(x18)  # Residual connection
        x20 = self.conv20(x19)  # Residual connection
        x21 = self.conv21(x20)  # Residual connection

        # Combine features with residual connections
        x15_1 = x21 + x4 + x1  # Residual connection

        x16 = self.conv16(x15_1)

        temp = self.upsample(x16, scale=scale)
        x17 = self.conv17(temp)
        out = self.add_mean(x17)

        return out