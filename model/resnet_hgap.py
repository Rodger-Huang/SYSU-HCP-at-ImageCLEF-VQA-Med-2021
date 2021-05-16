import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class resnet50HGap(nn.Module):
    def __init__(self, pretrained=True, num_classes=330):
        super(resnet50HGap, self).__init__()
        self.resnet50 = resnet50(pretrained=pretrained)
        self.model_list = list(self.resnet50.children())
        self.conv1 = self.model_list[0]
        self.bn1 = self.model_list[1]
        self.relu = self.model_list[2]
        self.maxpool = self.model_list[3]

        self.layer1 = self.model_list[4]
        self.layer2 = self.model_list[5]
        self.layer3 = self.model_list[6]
        self.layer4 = self.model_list[7]

        self.classifier = nn.Linear(3968, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        p1 = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        p2 = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        x = self.layer1(x)
        p3 = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.layer2(x)
        p4 = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.layer3(x)
        p5 = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.layer4(x)
        p6 = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        out = torch.cat([p1, p2, p3, p4, p5, p6], 1)
        out = self.classifier(out)
        return out


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


if __name__ == '__main__':
    model = resnet50HGap(pretrained=True)
    model(torch.rand(2, 3, 224, 224))

