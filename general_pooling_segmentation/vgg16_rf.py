import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from collections import OrderedDict

class VGG(nn.Module):
    def __init__(self, num_classes = 21):
        super(VGG, self).__init__()
        self.features = nn.Sequential(OrderedDict([\
	('conv1_1',nn.Conv2d(3,64,3, padding=1)),\
	('relu1_1',nn.ReLU(inplace=True)),\
	('conv1_2',nn.Conv2d(64,64,3, padding=1)),\
	('relu1_2',nn.ReLU(inplace=True)),\
	('pool1',nn.MaxPool2d(2, stride=2, padding=1)),\
	('conv2_1',nn.Conv2d(64,128,3, padding=1)),\
	('relu2_1',nn.ReLU(inplace=True)),\
	('conv2_2',nn.Conv2d(128,128,3, padding=1)),\
	('relu2_2',nn.ReLU(inplace=True)),\
	('pool2',nn.MaxPool2d(2, stride=2, padding=1)),\
	('conv3_1',nn.Conv2d(128,256,3, padding=1)),\
	('relu3_1',nn.ReLU(inplace=True)),\
	('conv3_2',nn.Conv2d(256,256,3, padding=1)),\
	('relu3_2',nn.ReLU(inplace=True)),\
	('conv3_3',nn.Conv2d(256,256,3, padding=1)),\
	('relu3_3',nn.ReLU(inplace=True)),\
	('pool3',nn.MaxPool2d(2, stride=2, padding=1)),\
	('conv4_1',nn.Conv2d(256,512,3, padding=1)),\
	('relu4_1',nn.ReLU(inplace=True)),\
	('conv4_2',nn.Conv2d(512,512,3, padding=1)),\
	('relu4_2',nn.ReLU(inplace=True)),\
	('conv4_3',nn.Conv2d(512,512,3, padding=1)),\
	('relu4_3',nn.ReLU(inplace=True)),\
	('pool4',nn.MaxPool2d(2, stride=1, padding=0)),\
	('conv5_1',nn.Conv2d(512,512,3, padding=2, dilation=2)),\
	('relu5_1',nn.ReLU(inplace=True)),\
	('conv5_2',nn.Conv2d(512,512,3, padding=2, dilation=2)),\
	('relu5_2',nn.ReLU(inplace=True)),\
	('conv5_3',nn.Conv2d(512,512,3, padding=2, dilation=2)),\
	('relu5_3',nn.ReLU(inplace=True)),\
	('pool5',nn.MaxPool2d(3, stride=1, padding=1)),\
	('fc6',nn.Conv2d(512,1024,3, padding=1, dilation=1)),\
	('relu6',nn.ReLU(inplace=True)),\
	('drop6',nn.Dropout2d(p=0.5)),\
	('fc7',nn.Conv2d(1024,1024,1)),\
	('relu7',nn.ReLU(inplace=True)),\
	('drop7',nn.Dropout2d(p=0.5)),\
	('fc8_voc',nn.Conv2d(1024,num_classes,1)),\
    #('upsample',  nn.UpsamplingBilinear2d(size=(224,224))),\
 ]))
		

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        #x = self.classifier(x)
        return x 




   






            
    
             
       

      
