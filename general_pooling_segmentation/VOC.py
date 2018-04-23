import os
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

num_classes = 21
ignore_label = 255
root = '/home/zhangjingyi/res/mixed-gated-pooling/general_pooling_segmentation/VOC2012-May-11-data/VOCdevkit/VOC2012'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        data_list = [l.strip('\n').split()[0] for l in open(os.path.join(root, 'ImageSets', 'Segmentation', 'deeplab_trainval_aug.txt')).readlines()]
        mask_list = [l.strip('\n').split()[1] for l in open(os.path.join(root, 'ImageSets', 'Segmentation', 'deeplab_trainval_aug.txt')).readlines()]
        for i in range(len(data_list)):
            item = (root+data_list[i], root+mask_list[i])
            items.append(item)
    elif mode == 'val':
        data_list = [l.strip('\n').split()[0] for l in open(os.path.join(root, 'ImageSets', 'Segmentation', 'deeplab_val.txt')).readlines()]
        mask_list = [l.strip('\n').split()[1] for l in open(os.path.join(root, 'ImageSets', 'Segmentation', 'deeplab_val.txt')).readlines()]
        for i in range(len(data_list)):
            item = (root+data_list[i], root+mask_list[i])
            items.append(item)
    # else:
    #     img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
    #     data_list = [l.strip('\n') for l in open(os.path.join(
    #         root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
    #     for it in data_list:
    #         items.append((img_path, it))
    return items


class VOC(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
    
        mask = Image.open(mask_path)

        

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)*255
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)
make_dataset('train')