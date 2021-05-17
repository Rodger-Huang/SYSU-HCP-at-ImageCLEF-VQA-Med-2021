import os
import socket
import argparse
from datetime import datetime
import glob
from tqdm import tqdm
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
import wandb

from dataloaders import tnsc_dataset
from dataloaders import custom_transforms as trforms
import utils
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
# import pretrainedmodels
from model import pretrainedmodels


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-world_size', type=int, default=1)
    parser.add_argument('-port', type=str, default='12356')
    # wandb setting
    parser.add_argument('-dryrun', action="store_true")
    # dataset setting
    parser.add_argument('-train_set', type=str, default='train')
    # Model setting
    parser.add_argument('-backbone', type=str, default='resnest50')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-pretrain', type=str, default='')

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-classes', type=int, default=330)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-nepochs', type=int, default=100)
    parser.add_argument('-resume_path', type=str, default='')

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=20)

    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=25)

    # tricks
    parser.add_argument('--mixup', action='store_true', 
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label_smooth', action='store_true', 
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--superloss', action='store_true',)
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='whether train the model with warmup. default is false.')
    parser.add_argument('--warmup_epoch', action='store_true', default=5,
                        help='when to train the model with warmup. default is 5.')

    return parser.parse_args()

def print0(string):
    if dist.get_rank() == 0:
        print(string)

def main(rank, args):
    if not args.dryrun and rank == 0:
        name = '{}_{}_{}'.format(args.backbone, args.input_size, args.train_set)
        if args.mixup:
            name += '_mixup'
        if args.label_smooth:
            name += '_labelSmooth'
        if args.superloss:
            name += '_SuperLoss'
        wandb.init(project='MVQA', config=args, name=name)
        args = wandb.config
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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
            args.backbone == 'vgg19HGap' or args.backbone == 'vgg16HGap_resizer' or \
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

    if args.resume_path is '':
        resume_epoch = 0
        print0('Training from scratch...')
    else:
        resume = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
        state_dict = backbone.state_dict()
        for k in state_dict:
            k_m = 'module.' + k
            state_dict[k] = resume['state_dict'][k_m]
        backbone.load_state_dict(state_dict)
        resume_epoch = resume['epoch'] + 1
        run_id = resume['run_id']
        print0('Initializing weights from: {}, epoch: {}...'.format(args.resume_path, resume_epoch))

    if args.pretrain is not '':
        backbone = utils.load_pretrain_model(backbone, torch.load(args.pretrain)['state_dict'])
    
    if rank == 0:
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        
        if args.resume_path is '':
            runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
            run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, 'run', 'run_{}_{}'.format(name, run_id))
        log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%M%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

        logger = open(os.path.join(save_dir, 'log.txt'), 'w')
        logger.write('optim: SGD \nlr=%.4f\nweight_decay=%.4f\nmomentum=%.4f\nupdate_lr_every=%d\nseed=%d\n' %
                    (args.lr, args.weight_decay, args.momentum, args.update_lr_every, args.seed))

        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))

    backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    backbone.to(rank)
    backbone = DDP(backbone, device_ids=[rank], find_unused_parameters=True)

    backbone_optim = optim.SGD(
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    if args.resume_path is not '':
        if (resume_epoch - 1) % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, (resume_epoch - 1), args.nepochs, 0.9)
            print0('(poly lr policy) learning rate: {}'.format(lr_))
            backbone_optim = optim.SGD(
                backbone.parameters(),
                lr=lr_,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size + 8, args.input_size + 8)),
        trforms.RandomCrop(size=(args.input_size, args.input_size)),
        # trforms.colorjitter_sample(parameters=(0.2, 0.2, 0.2, 0.)),
        trforms.RandomHorizontalFlip(),
        # trforms.Grayscale(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # trforms.Normalize(mean=(0.2565, 0.2564, 0.2564), std=(0.2223, 0.2224, 0.2222)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        # trforms.Grayscale(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # trforms.Normalize(mean=(0.2565, 0.2564, 0.2564), std=(0.2223, 0.2224, 0.2222)),
        trforms.ToTensor()])

    trainset = tnsc_dataset.MedLTDataset(mode='train', path=args.train_set, transform=composed_transforms_tr, return_size=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8, sampler=train_sampler)

    valset = tnsc_dataset.MedLTDataset(mode='val', path='val', transform=composed_transforms_ts, return_size=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    valloader = DataLoader(valset, batch_size=1, num_workers=2, sampler=val_sampler)

    num_iter_tr = len(trainloader)
    nitrs = resume_epoch * num_iter_tr
    nsamples = args.batch_size * nitrs
    print0('each_epoch_num_iter: %d' % (num_iter_tr))

    global_step = 0
    best_acc = 0
    best_acc_pp = 0

    recent_losses = []
    print0('Training Network')
    if not args.dryrun and rank == 0:
        wandb.watch(backbone)

    for epoch in range(resume_epoch, args.nepochs):
        backbone.train()
        epoch_losses = []

        train_pbar = tqdm(trainloader, unit_scale=args.batch_size * args.world_size, disable=not(rank==0))
        for ii, sample_batched in enumerate(train_pbar):
            img, label = sample_batched['image'].to(rank), sample_batched['label'].to(rank)
            global_step += args.batch_size

            if args.mixup:
                inputs, targets_a, targets_b, lam = utils.mixup_data(img, label, args.mixup_alpha, use_cuda=True, rank=rank)
                loss_func = utils.mixup_criterion(targets_a, targets_b, lam)
                outputs = backbone(inputs)
                
                if args.label_smooth:
                    if args.superloss:
                        criterion = utils.LabelSmoothingCrossEntropyWithSuperLoss(classes=args.classes, rank=rank)
                    else:
                        criterion = utils.LabelSmoothingCrossEntropy(classes=args.classes)
                    # criterion = utils.AngleLoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                loss = loss_func(criterion, outputs)

            else:
                feats = backbone(img)
                loss = utils.CELoss(logit=feats, target=label, reduction='mean')

            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss
            
            backbone_optim.zero_grad()
            loss.backward()
            backbone_optim.step()

            nitrs += 1
            nsamples += args.batch_size

            if nitrs % args.log_every == 0 and rank == 0:
                meanloss = sum(recent_losses) / len(recent_losses)

                train_pbar.set_description('epoch: %d ii: %d trainloss: %.2f ' % (
                    epoch, ii, meanloss))
                writer.add_scalar('data/trainloss', meanloss, nsamples)
                # wandb.log({"trainloss": meanloss})

            if ii % (num_iter_tr // 10) == 0 and rank == 0:
                grid_image = make_grid(img[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                # wandb.log({"Image": wandb.Image(grid_image)})

        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print0('(poly lr policy) learning rate: {}'.format(lr_))
            backbone_optim = optim.SGD(
                params=backbone.parameters(),
                lr=lr_,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        # validation
        backbone.eval()
        acc = torch.zeros(1).to(rank)
        acc_pp = torch.zeros(1).to(rank)

        val_pbar = tqdm(valloader, unit_scale=1 * args.world_size, disable=not(rank==0))
        for ii, sample_batched in enumerate(val_pbar):
            img, label = sample_batched['image'].to(rank), sample_batched['label'].to(rank)

            feats = backbone(img)
            # test time augmentation
            img_flip = flip_lr(img)
            feats_flip = backbone(img_flip)
            feats_pp = (feats + feats_flip) / 2.0
            if torch.argmax(feats, dim=1, keepdim=False) == label:
                acc += 1
            if torch.argmax(feats_pp, dim=1, keepdim=False) == label:
                acc_pp += 1

        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_pp, op=dist.ReduceOp.SUM)
        acc /= len(valset)
        acc_pp /= len(valset)

        if(acc > best_acc or acc_pp > best_acc_pp) and rank == 0:
            best_acc = acc if acc > best_acc else best_acc
            best_acc_pp = acc_pp if acc_pp > best_acc_pp else best_acc_pp
            backbone_save_path = os.path.join(save_dir, 'models', 
                'best_backbone_e{}_{:.4f}_{:.4f}.pth'.format(epoch, acc.item(), acc_pp.item()))
            torch.save({
                'epoch': epoch,
                'run_id': run_id,
                'state_dict': backbone.state_dict(),
            }, backbone_save_path)
            print("Save best backbone at {}\n".format(backbone_save_path))

        print0('Validation:')
        print0('epoch: %d, images: %d, acc: %.4f, acc_pp: %.4f' % (epoch, len(valset), acc, acc_pp))
        if rank == 0:
            logger.write('epoch: %d, images: %d acc: %.4f' % (epoch, len(valset), acc.item()))

        if not args.dryrun and rank == 0:
            # writer.add_scalar('data/valid_acc', acc.item(), nsamples)
            wandb.log({"valid_acc": acc.item(), 
                "valid_acc_pp": acc_pp.item()})

        if epoch % args.save_every == args.save_every - 1 and rank == 0:
            backbone_save_path = os.path.join(save_dir, 'models', 
                'backbone_epoch-{}_{:.4f}_{:.4f}.pth'.format(epoch, acc.item(), acc_pp.item()))
            torch.save({
                'epoch': epoch,
                'run_id': run_id,
                'state_dict': backbone.state_dict(),
            }, backbone_save_path)
            print("Save backbone at {}\n".format(backbone_save_path))

    if rank == 0:
        writer.close()
        if not args.dryrun:
            wandb.finish()


if __name__ == '__main__':
    args = get_arguments()
    if args.dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    mp.spawn(main,
        args=(args,),
        nprocs=args.world_size,
        join=True)

