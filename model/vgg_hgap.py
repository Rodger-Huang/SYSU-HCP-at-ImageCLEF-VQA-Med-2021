import torch
import torch.nn as nn
import torchvision
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import LSoftmaxLinear
from utils import AngleLinear
from block.NonLocal import NonLocalBlock
from block.resizerNetwork import ResizerNetwork

class vgg16HGap(nn.Module):
    def __init__(self, pretrained=True, num_classes=330, rank=None, margin=None):
        super(vgg16HGap, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:16])
        self.conv4 = nn.Sequential(*self.model_list[16:23])
        self.conv5 = nn.Sequential(*self.model_list[23:30])
        self.conv6 = nn.Sequential(self.model_list[30])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        self.pool6 = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Linear(1984, num_classes)
        
        # self.lsoftmax_linear = LSoftmaxLinear(
        #     input_features=1984, output_features=num_classes, margin=margin, device=rank)
        
        # self.asoftmax = AngleLinear(1984, num_classes, m=margin)

    def forward(self, x, targeta=None, targetb=None):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        p6 = self.pool6(y6)
        
        f = torch.cat([p1, p2, p3, p4, p5, p6], 1).squeeze(3).squeeze(2)
        
        return self.classifier(f)
        
        # if self.training:
        #     outa = self.lsoftmax_linear(input=f, target=targeta)
        #     outb = self.lsoftmax_linear(input=f, target=targetb)
        #     return outa, outb
        # else:
        #     return self.lsoftmax_linear(input=f)

        # return self.asoftmax(f)


class vgg16HGap_bn(nn.Module):
    def __init__(self, pretrained=True, num_classes=330):
        super(vgg16HGap_bn, self).__init__()
        self.vgg = torchvision.models.vgg16_bn(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:6])
        self.conv2 = nn.Sequential(*self.model_list[6:13])
        self.conv3 = nn.Sequential(*self.model_list[13:23])
        self.conv4 = nn.Sequential(*self.model_list[23:33])
        self.conv5 = nn.Sequential(*self.model_list[33:43])
        self.conv6 = nn.Sequential(self.model_list[43])
        self.pool1 = nn.AvgPool2d(224)
        self.pool2 = nn.AvgPool2d(112)
        self.pool3 = nn.AvgPool2d(56)
        self.pool4 = nn.AvgPool2d(28)
        self.pool5 = nn.AvgPool2d(14)
        self.pool6 = nn.AvgPool2d(7)
        self.classifier = nn.Linear(1984, num_classes)
        
    def forward(self, x):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        p6 = self.pool6(y6)
        
        out = torch.cat([p1,p2,p3,p4,p5,p6], 1).squeeze(3)
        out = out.squeeze(2)
        out = self.classifier(out)
        return out


class vgg16_multi_scales(nn.Module):
    def __init__(self, pretrained=True, num_classes=330):
        super(vgg16_multi_scales, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:16])
        self.conv4 = nn.Sequential(*self.model_list[16:23])
        self.conv5 = nn.Sequential(*self.model_list[23:30])
        self.conv6 = nn.Sequential(self.model_list[30])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        self.pool6 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        p6 = self.pool6(y6)
        
        out1 = p3.squeeze(3).squeeze(2)
        out1 = self.fc1(out1)
        out2 = p4.squeeze(3).squeeze(2)
        out2 = self.fc2(out2)
        out3 = p5.squeeze(3).squeeze(2)
        out3 = self.fc3(out3)
        out4 = p6.squeeze(3).squeeze(2)
        out4 = self.fc4(out4)
        if self.training:
            return [out1, out2, out3, out4]
        else:
            return out4

class vgg16_add(nn.Module):
    def __init__(self, pretrained=True, num_classes=330):
        super(vgg16_add, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:16])
        self.conv4 = nn.Sequential(*self.model_list[16:23])
        self.conv5 = nn.Sequential(*self.model_list[23:30])
        self.conv6 = nn.Sequential(self.model_list[30])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        self.pool6 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        p6 = self.pool6(y6)
        
        out1 = p3.squeeze(3).squeeze(2)
        out1 = self.fc1(out1)
        out2 = p4.squeeze(3).squeeze(2)
        out2 = self.fc2(out2)
        out3 = p5.squeeze(3).squeeze(2)
        out3 = self.fc3(out3)
        out4 = p6.squeeze(3).squeeze(2)
        out4 = self.fc4(out4)

        return out1 + out2 + out3 + out4

class vgg19HGap(nn.Module):
    def __init__(self, pretrained=True, num_classes=330):
        super(vgg19HGap, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:18])
        self.conv4 = nn.Sequential(*self.model_list[18:27])
        self.conv5 = nn.Sequential(*self.model_list[27:36])
        self.conv6 = nn.Sequential(self.model_list[36])
        self.pool1 = nn.AvgPool2d(224)
        self.pool2 = nn.AvgPool2d(112)
        self.pool3 = nn.AvgPool2d(56)
        self.pool4 = nn.AvgPool2d(28)
        self.pool5 = nn.AvgPool2d(14)
        self.pool6 = nn.AvgPool2d(7)
        self.classifier = nn.Linear(1984, num_classes)
        
    def forward(self, x):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        p6 = self.pool6(y6)
        
        out = torch.cat([p1,p2,p3,p4,p5,p6],1).squeeze(3)
        out = out.squeeze(2)
        out = self.classifier(out)
        return out


class vgg16NonLocal(nn.Module):
    def __init__(self, pretrained=True, num_classes=330, rank=None, margin=None):
        super(vgg16NonLocal, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:16])
        self.conv4 = nn.Sequential(*self.model_list[16:23])
        self.conv5 = nn.Sequential(*self.model_list[23:30])
        self.conv6 = nn.Sequential(self.model_list[30])
        self.pool1 = nn.AvgPool2d(224)
        self.pool2 = nn.AvgPool2d(112)
        self.pool3 = nn.AvgPool2d(56)
        self.pool4 = nn.AvgPool2d(28)
        self.pool5 = nn.AvgPool2d(14)
        self.pool6 = nn.AvgPool2d(7)
        
        self.non4 = NonLocalBlock(512)
        self.non5 = NonLocalBlock(512)
        self.non6 = NonLocalBlock(512)
        
        self.classifier = nn.Linear(1984, num_classes)

    def forward(self, x, targeta=None, targetb=None):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        y4 = self.non4(y4)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        y5= self.non5(y5)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        y6 = self.non6(y6)
        p6 = self.pool6(y6)
        
        f = torch.cat([p1, p2, p3, p4, p5, p6], 1).squeeze(3).squeeze(2)
        
        return self.classifier(f)


class vgg16HGap_resizer(nn.Module):
    def __init__(self, pretrained=True, num_classes=330, rank=None, margin=None, output_size=(224, 224), num_residuals=1):
        super(vgg16HGap_resizer, self).__init__()
        self.resizer = ResizerNetwork(output_size=output_size, num_residuals=num_residuals)
        self.vgg = torchvision.models.vgg16(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:16])
        self.conv4 = nn.Sequential(*self.model_list[16:23])
        self.conv5 = nn.Sequential(*self.model_list[23:30])
        self.conv6 = nn.Sequential(self.model_list[30])
        self.pool1 = nn.AvgPool2d(224)
        self.pool2 = nn.AvgPool2d(112)
        self.pool3 = nn.AvgPool2d(56)
        self.pool4 = nn.AvgPool2d(28)
        self.pool5 = nn.AvgPool2d(14)
        self.pool6 = nn.AvgPool2d(7)
        
        self.classifier = nn.Linear(1984, num_classes)

    def forward(self, x):
        x = self.resizer(x)
        y1 = self.conv1(x)
        p1 = self.pool1(y1)
        
        y2 = self.conv2(y1)
        p2 = self.pool2(y2)
        
        y3 = self.conv3(y2)
        p3 = self.pool3(y3)
        
        y4 = self.conv4(y3)
        p4 = self.pool4(y4)
        
        y5 = self.conv5(y4)
        p5 = self.pool5(y5)
        
        y6 = self.conv6(y5)
        p6 = self.pool6(y6)
        
        f = torch.cat([p1, p2, p3, p4, p5, p6], 1).squeeze(3).squeeze(2)
        
        return self.classifier(f)


if __name__ == '__main__':
    model = vgg16HGap_resizer(pretrained=True)
    model(torch.rand(2, 3, 512, 512))

