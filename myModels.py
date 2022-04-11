

# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
from torchvision.models import resnet50

"""
a = resnet50(pretrained=True)
in_features = a.fc.in_features
a.fc = nn.Linear(in_features, out_features=10)
"""




class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1), #32 32 6
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1) # b, -1
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape) # b, xxxx
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        # b, 10
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.conv1(x2)
        x4 = x3 + x1
        out = self.relu(x4)
        return out

        pass

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3)
        self.layer1 = residual_block(in_channels=64)
        self.layer2 = residual_block(in_channels=128)
        self.layer3 = residual_block(in_channels=256)
        self.expand1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.expand2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128,kernel_size=3, padding=1),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(128*16*16, 1024), nn.Dropout(0.2),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024,500),nn.Dropout(0.2), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(500,84), nn.Dropout(0.2),nn.ReLU())
        self.fc4 = nn.Linear(84,num_out)
        
        

        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        pass
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)
        x = self.stem_conv(x)
        x = self.layer1(x)
        x = self.layer1(x)
        x = self.layer1(x)
        x = self.layer1(x)
        x = self.expand1(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x =self.expand2(x)
        x = self.layer3(x)
        x = self.layer3(x)
        x = self.layer3(x)
        x = self.cnn1(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1) # b, -1

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        out = x
        return out

        pass
