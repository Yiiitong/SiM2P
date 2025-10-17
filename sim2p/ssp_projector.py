import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d_block(in_ch, out_ch, kernel_size=3, stride=(1,1,1), padding=(1,1,1)):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv3d_block(channels, channels, kernel_size=3, stride=(1,1,1))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        x = self.relu(x)
        return x

class DownBlockZ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_init = conv3d_block(in_ch, out_ch, stride=(1,1,1))
        self.res       = ResidualBlock3D(out_ch)
        self.down      = conv3d_block(out_ch, out_ch,
                                      kernel_size=(3,3,3),
                                      stride=(1,1,2),  # Only halves the last dim
                                      padding=(1,1,1))

    def forward(self, x):
        skip = x  # save skip connection
        x = self.conv_init(x)
        x = self.res(x)
        
        x_down = self.down(x)  # halve the last dimension
        return skip, x_down


class ProjectorNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Encoder
        self.enc1 = DownBlockZ(in_ch,    base)     
        self.enc2 = DownBlockZ(base,     base*2)
        self.enc3 = DownBlockZ(base*2,   base*4)

        self.enc4 = DownBlockZ(base*4,   base*2)
        self.enc5 = DownBlockZ(base*2,   base)

        # Final 1x1x1 conv
        self.final_conv = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):

        x = F.interpolate(
            x,
            size=(x.size(2), x.size(3), 64),  # (D=80, H=80, W=64)
            mode='trilinear',
            align_corners=False
        )

        s1, x1 = self.enc1(x)
        s2, x2 = self.enc2(x1)
        s3, x3 = self.enc3(x2)

        s4, x4 = self.enc4(x3)
        s5, x5 = self.enc5(x4)

        out = self.final_conv(x5)
        
        return out

