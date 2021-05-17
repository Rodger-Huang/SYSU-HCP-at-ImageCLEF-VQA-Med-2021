import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import csv
from dataloaders import custom_transforms as trforms
from dataloaders import tnsc_dataset
from utils import flip_lr

from torchvision.models.resnet import resnet34, resnet18, resnet50, resnet101
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.densenet import densenet169, densenet121, densenet201
from resnest.torch import resnest50, resnest101
from model.vgg_hgap import vgg16HGap, vgg16HGap_bn, vgg19HGap, vgg16_multi_scales, \
    vgg16_add
from model.resnet_hgap import resnet50HGap
from model.resnest_hgap import resnest50HGap
from model.repvgg import get_RepVGG_func_by_name
# import pretrainedmodels
from model import pretrainedmodels


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-backbone', type=str, default='resnest50')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-model_path', type=str, default='path2model')
    parser.add_argument('-classes', type=int, default=330)
    parser.add_argument('-csv_path', type=str, default='')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.backbone == 'resnet18' or args.backbone == 'resnet34' or \
            args.backbone == 'resnet50' or args.backbone == 'resnet101' or \
            args.backbone == 'resnest50' or args.backbone == 'resnest101':
        backbone = eval(args.backbone)(pretrained=True)
        backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=args.classes)
        # # The following operation is not working. The output classes are still 1000.
        # backbone.fc.out_features = args.classes
    elif args.backbone == 'vgg16' or args.backbone == 'vgg16_bn' or \
            args.backbone == 'vgg19' or args.backbone == 'vgg19_bn':
        backbone = eval(args.backbone)(pretrained=True)
        backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, args.classes),
        )
    elif args.backbone == 'vgg16HGap' or args.backbone == 'vgg16HGap_bn' or \
            args.backbone == 'vgg19HGap' or \
            args.backbone == 'vgg16_multi_scales' or \
            args.backbone == 'vgg16_add' or \
            args.backbone == 'resnet50HGap' or \
            args.backbone == 'resnest50HGap':
        backbone = eval(args.backbone)(pretrained=True, num_classes=args.classes)
    elif args.backbone == 'densenet121' or args.backbone == 'densenet169' or \
            args.backbone == 'densenet201':
        backbone = eval(args.backbone)(pretrained=True)
        backbone.classifier = nn.Linear(in_features=backbone.classifier.in_features, out_features=args.classes)
    elif args.backbone == 'RepVGG-B0' or args.backbone == 'RepVGG-B1' or args.backbone == 'RepVGG-B2':
        backbone = get_RepVGG_func_by_name(args.backbone)(deploy=False)
        backbone.load_state_dict(torch.load('checkpoint/RepVGG/{}-train.pth'.format(args.backbone)))
        backbone.linear = nn.Linear(in_features=backbone.linear.in_features, out_features=args.classes)
    elif args.backbone == 'inceptionresnetv2' or args.backbone == 'inceptionv4' or \
            args.backbone == 'xception' or args.backbone == 'polynet' or \
            args.backbone == 'se_resnet50':
        # backbone = pretrainedmodels.__dict__[args.backbone](pretrained='imagenet')
        backbone = eval('pretrainedmodels.' + args.backbone)(pretrained='imagenet')
        backbone.last_linear = nn.Linear(in_features=backbone.last_linear.in_features, out_features=args.classes, bias=True)
    else:
        raise NotImplementedError
    
    model = torch.load(args.model_path, map_location='cpu')
    state_dict = backbone.state_dict()
    for k in state_dict:
        k_m = 'module.' + k
        state_dict[k] = model['state_dict'][k_m]
    backbone.load_state_dict(state_dict)
    torch.cuda.set_device(device=0)
    backbone.cuda()
    backbone.eval()

    composed_transforms_val = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    valset = tnsc_dataset.MedLTDataset(mode='val', path='val', transform=composed_transforms_val, return_size=False)
    valloader = DataLoader(valset, batch_size=1, num_workers=2, shuffle=False)

    acc = 0
    num_category = torch.zeros(args.classes)
    num_right = torch.zeros(args.classes)
    acc_pp = 0
    num_right_pp = torch.zeros(args.classes)
    wrong_name = [[] for _ in range(args.classes)]

    val_pbar = tqdm(valloader, unit_scale=1)
    for ii, sample_batched in enumerate(val_pbar):
        img, label = sample_batched['image'].cuda(), sample_batched['label'].cuda()
        feats = backbone.forward(img)

        img_flip = flip_lr(img)
        feats_flip = backbone.forward(img_flip)
        feats_pp = (feats + feats_flip) / 2

        num_category[label] += 1
        if torch.argmax(feats, dim=1, keepdim=False) == label:
            acc += 1
            num_right[label] += 1
        else:
            wrong_name[label].append(sample_batched['label_name'][0])
    
        if torch.argmax(feats_pp, dim=1, keepdim=False) == label:
            acc_pp += 1
            num_right_pp[label] += 1

    acc /= len(valset)
    acc_pp /= len(valset)
    record_path = args.csv_path + 'epoch' + str(model['epoch']) + "_" + str(acc) + "_" + str(acc_pp) + '.csv'
    f = open(record_path, 'w')
    writer = csv.writer(f)
    writer.writerow(('category', 'number', 'right', 'right_pp', 'percentage', 'wrong_name'))
    for i in range(args.classes):
        writer.writerow((i, num_category[i].item(), num_right[i].item(), num_right_pp[i].item(), \
            '{:.4f}, {:.4f}'.format((num_right[i] / num_category[i]).item(), (num_right_pp[i] / num_category[i]).item()), 
            wrong_name[i]))
    f.close()
    print("acc: {:.4f}, acc_pp: {:.4f}".format(acc, acc_pp))

if __name__ == '__main__':
    args = get_arguments()
    main(args)
