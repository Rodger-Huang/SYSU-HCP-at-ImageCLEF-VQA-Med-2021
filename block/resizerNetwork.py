from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        return out


class ResizerNetwork(nn.Module):
    def __init__(self, output_size=(224, 224), num_residuals=1):
        super(ResizerNetwork, self).__init__()

        self.output_size = output_size

        self.conv1 = nn.Conv2d(3, 16, 7, stride=1, padding=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        blocks = OrderedDict([('res'+str(i+1), ResBlock()) for i in range(num_residuals)])
        self.residual_blocks = nn.Sequential(blocks)

        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 3, 7, stride=1, padding=3)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.lrelu(out)
        out = self.bn2(out)

        out = F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)

        residual_input = out

        out = self.residual_blocks(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual_input

        out = self.conv4(out)

        out += F.interpolate(identity, size=self.output_size, mode='bilinear', align_corners=False)

        return out
