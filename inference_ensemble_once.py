import argparse
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
from dataloaders import custom_transforms as trforms
from dataloaders import tnsc_dataset
from utils import flip_lr

from torchvision.models.resnet import resnet34, resnet18, resnet50, resnet101
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.densenet import densenet169, densenet121, densenet201
from resnest.torch import resnest50, resnest101
from model.vgg_hgap import vgg16HGap, vgg16HGap_bn, vgg19HGap, vgg16_multi_scales, \
    vgg16_add, vgg16HGap_resizer
from model.resnet_hgap import resnet50HGap
from model.resnest_hgap import resnest50HGap
from model.repvgg import get_RepVGG_func_by_name
from model import pretrainedmodels


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-backbone', type=str, default='resnest50')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-model_path', type=str, default='path2model')
    parser.add_argument('-classes', type=int, default=330)

    return parser.parse_args()

modal_mask = json.load(open('modelmask.json', 'r'))
modalties = modal_mask.keys()

def main(args, test_pbar):
    backbone, input_size, model_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'

    if backbone == 'resnet18' or backbone == 'resnet34' or \
            backbone == 'resnet50' or backbone == 'resnet101' or \
            backbone == 'resnest50' or backbone == 'resnest101':
        backbone = eval(backbone)(pretrained=True)
        backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=330)
    elif backbone == 'vgg16' or backbone == 'vgg16_bn' or \
            backbone == 'vgg19' or backbone == 'vgg19_bn':
        backbone = eval(backbone)(pretrained=True)
        backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 330),
        )
    elif backbone == 'vgg16HGap' or backbone == 'vgg16HGap_bn' or \
            backbone == 'vgg19HGap' or backbone == 'vgg16HGap_resizer' or \
            backbone == 'vgg16_multi_scales' or \
            backbone == 'vgg16_add' or \
            backbone == 'resnet50HGap' or \
            backbone == 'resnest50HGap':
        backbone = eval(backbone)(pretrained=True, num_classes=330)
    elif backbone == 'densenet121' or backbone == 'densenet169' or \
            backbone == 'densenet201':
        backbone = eval(backbone)(pretrained=True)
        backbone.classifier = nn.Linear(in_features=backbone.classifier.in_features, out_features=330)
    elif backbone == 'RepVGG-B0' or backbone == 'RepVGG-B1' or backbone == 'RepVGG-B2':
        backbone = get_RepVGG_func_by_name(backbone)(deploy=False)
        backbone.load_state_dict(torch.load('checkpoint/RepVGG/{}-train.pth'.format(backbone)))
        backbone.linear = nn.Linear(in_features=backbone.linear.in_features, out_features=330)
    elif backbone == 'inceptionresnetv2' or backbone == 'inceptionv4' or \
            backbone == 'xception' or backbone == 'polynet' or \
            backbone == 'se_resnet50':
        backbone = eval('pretrainedmodels.' + backbone)(pretrained='imagenet')
        backbone.last_linear = nn.Linear(in_features=backbone.last_linear.in_features, out_features=330, bias=True)
    else:
        raise NotImplementedError

    model = torch.load(model_path, map_location='cpu')
    state_dict = backbone.state_dict()
    for k in state_dict:
        k_m = k
        if not model['state_dict'].__contains__(k_m):
            k_m = 'module.' + k
        state_dict[k] = model['state_dict'][k_m]
    backbone.load_state_dict(state_dict)
    torch.cuda.set_device(device=0)
    backbone.cuda()
    backbone.eval()

    pred_list = []
    for sample_batched in test_pbar:
        q = sample_batched['question'][0]
        mask = []
        for modality in modalties:
            if modality in q:
                mask = modal_mask[modality]

        img = sample_batched['image'].cuda()
        feats = backbone(img)
        prob, pred_idx = torch.topk(feats, 20, dim=1)

        if prob[0][0].item() > 0.5:
            pred = pred_idx[0][0]
            pred_list.append(pred.item())
            continue

        if len(mask) != 0:
            for i in range(len(pred_idx[0])):
                if pred_idx[0][i] in mask:
                    pred = i
                    break
                if i == 7:
                    pred = pred_idx[0][0]

        else:
            pred = pred_idx[0][0]

        # print(pred)
        # img_flip = flip_lr(img)
        # feats_flip = backbone(img_flip)
        # feats_pp = (feats + feats_flip) / 2
        # pred_pp = torch.argmax(feats_pp, dim=1, keepdim=False)
        pred_list.append(pred.item())
    return pred_list


if __name__ == '__main__':
    arg_list = [
        ('resnet50', 256, 'VQA-MED-2021-Models/res50/models/backbone_epoch-49_1.0000_1.0000.pth'),
        ('resnet50HGap', 256, 'VQA-MED-2021-Models/res50hgap/models/backbone_epoch-49_1.0000_1.0000.pth'),
        ('resnest50', 256, 'VQA-MED-2021-Models/ress50/models/backbone_epoch-29_0.9980_0.9980.pth'),
        ('resnest50HGap', 256, 'VQA-MED-2021-Models/ress50hgap/models/backbone_epoch-29_1.0000_1.0000.pth'),
        ('vgg16', 224, 'VQA-MED-2021-Models/vgg16/models/backbone_epoch-49_1.0000_1.0000.pth'),
        ('vgg16HGap', 224, 'VQA-MED-2021-Models/vgg16hgap/models/backbone_epoch-49_1.0000_1.0000.pth'),
        ('vgg19', 224, 'VQA-MED-2021-Models/vgg19/models/backbone_epoch-49_1.0000_1.0000.pth'),
        ('vgg19HGap', 224, 'VQA-MED-2021-Models/vgg19hgap/models/backbone_epoch-49_1.0000_1.0000.pth'),
    ]

    composed_transforms_test = transforms.Compose([
        trforms.FixedResizeI(size=(384, 384)),
        trforms.NormalizeI(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensorI()])

    testset = tnsc_dataset.MedLTDataset(mode='test', path='test2021', transform=composed_transforms_test, return_size=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    test_pbar = tqdm(testloader, unit_scale=1)

    result = []
    for arg in arg_list:
        preds = main(arg, test_pbar)
        result.append(preds)

    print(result)

    new_result = []
    for i in range(500):
        votes = []
        for res in result:
            votes.append(res[i])
        new_result.append(votes)

    ensemble_result = []
    for i in range(500):
        ensemble_result.append(max(new_result[i], key=new_result[i].count))

    with open("./res.txt", "a", newline='') as f:
        for i in ensemble_result:
            f.write(str(i)+'\n')

