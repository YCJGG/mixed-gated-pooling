import datetime
import os
import random
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils.transforms as extended_transforms
import VOC
from vgg16_rf import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from utils import joint_transforms
import time
import os
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data


img_path = '2007_000033.jpg'
mask_path = '2007_000033.png'
img = Image.open(img_path).convert('RGB')
img = np.array(img)
tmp = img[:,:,0]
img[:,:,0] = img[:,:,2]
img[:,:,2] = tmp
image=Image.fromarray(np.uint8(img)) 
mask = Image.open(mask_path)

mean_std = ([0.408, 0.457, 0.481], [1, 1, 1])

joint_transform_train = joint_transforms.Compose([
    joint_transforms.RandomCrop((321,321))
])

joint_transform_test = joint_transforms.Compose([
    joint_transforms.RandomCrop((512,512))
])

input_transform = standard_transforms.Compose([
    #standard_transforms.Resize((321,321)),
    #standard_transforms.RandomCrop(224),
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
target_transform = standard_transforms.Compose([
    #standard_transforms.Resize((224,224)),
    extended_transforms.MaskToTensor()
])

image = input_transform(image)*255
mask =  target_transform(mask)
img, mask = joint_transform_test(image, mask)
np.save('im.npy',img.numpy())