import os

import torch
from PIL import Image
from torch.utils import data


def make_dataset(root, mode):
    imgs = []

    with open(root + '/' + mode + '.txt', 'r') as f:
        lines = f.readlines()

    if mode == 'val':
        with open(root + '/label.txt', 'r') as f:
            lines2 = f.readlines()
    for i, line in enumerate(lines):
        name, category = line.strip().split(' ')
        if mode == 'val':
            _, question, _ = lines2[i].strip().split('|')
            imgs.append([name, category, question])
        else:
            imgs.append([name, category])

    return imgs


def make_testset(root, mode):
    imgs = []
    with open(root + '/' + mode + '/Task1-VQA-2021-TestSet-Questions.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        name, question = line.strip().split('|')
        imgs.append([name, question])
    return imgs


class MedLTDataset(data.Dataset):
    def __init__(self, mode='train', path='train', transform=None, return_size=False):
        self.mode = mode
        self.path = path
        self.transform = transform
        self.return_size = return_size

        root = 'data/'
        self.root = root
        if mode == 'train':
            imgs = make_dataset(root+mode, path)
        elif mode == 'val':
            imgs = make_dataset(root+mode, mode)
        elif mode == 'test':
            imgs = make_testset(root, path)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        question = None
        if not self.mode == 'test':
            if self.mode == 'train':
                image_name, label = self.imgs[item]
            elif self.mode == 'val':
                image_name, label, question = self.imgs[item]
            image_path = self.root + self.mode + '/images/' + image_name + '.jpg'
        else:
            image_name, question = self.imgs[item]
            image_path = self.root + self.path + '/images/' + image_name + '.jpg'
        
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))

        image = Image.open(image_path).convert('RGB')

        w, h = image.size
        size = (h, w)
        if not self.mode == 'test':
            sample = {'name': image_name, 'image': image, 'label': int(label)}
        else:
            sample = {'name': image_name, 'image': image}

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)

        sample['ID'] = image_name
        if question is not None:
            sample['question'] = question

        return sample

    def __len__(self):
        return len(self.imgs)
