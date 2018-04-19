import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

class VGG(nn.Module):
    def __init__(self, num_classes = 21):
        super(VGG, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3,64,3, padding=(100,100)),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),


            nn.Conv2d(256,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(512,512,3, padding=2, dilation = 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=2, dilation = 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=2,dilation = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512,1024,3,padding = 1),
            nn.ReLU (True),
            nn.Dropout(),
            nn.Conv2d(1024,1024,1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(1024,num_classes,1),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = self.classifier(x)
        return x 




   






            
    
             
       

      
