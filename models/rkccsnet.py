import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def _csm_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1/math.sqrt(m.weight.shape[-1]*m.weight.shape[-2]*m.weight.shape[-3]))

class block(nn.Module):
    def __init__(self,kernel_size):
        super(block,self).__init__()
        self.conv1=nn.Conv1d(16,16,kernel_size,1,padding=(kernel_size//2))
        self.conv2=nn.Conv1d(16,16,kernel_size,1,padding=(kernel_size//2))
        self.relu1=nn.LeakyReLU()
        self.relu2=nn.LeakyReLU()
    def forward(self,x):
        res=x
        x=self.relu1(self.conv1(x))
        x = self.conv2(x)
        x=res+x
        x=self.relu2(x)

        return x


class CSNet1(nn.Module):
    def __init__(self,sensing_rate):
        super(CSNet1, self).__init__()
        self.fcr = int(64 * sensing_rate)

        self.sample = nn.Sequential(nn.Conv1d(1, self.fcr, kernel_size=4, padding=0, stride=4, bias=False),
                                    nn.Conv1d(self.fcr, self.fcr, kernel_size=4, padding=0, stride=4, bias=False),
                                    nn.Conv1d(self.fcr, self.fcr, kernel_size=4, padding=0, stride=4, bias=False))
        '''self.sample=nn.Conv1d(1,self.fcr,kernel_size=64,padding=0,stride=64,bias=False)'''
        self.initial = nn.Conv1d(self.fcr, 64, kernel_size=1, padding=0, stride=1, bias=False)

        self.conv1=nn.Conv1d(1,16,11,1,padding=5)
        self.block1=block(11)
        self.block2=block(9)
        self.block3=block(7)
        self.block4=block(13)
        self.lstm=nn.LSTM(256,250)
        self.conv2=nn.Conv1d(64,1,11,1,padding=5)
        self.dense=nn.Linear(250,256)
        self.tan=nn.Tanh()
        self.relu=nn.LeakyReLU()

    def forward(self, input):
        output = self.sample(input)#1,2,4
        output=self.relu(self.initial(output))#1,64,4
        output=output.reshape(1,1,256)

        x=self.relu(self.conv1(output))
        x1=self.block1(x)
        x2=self.block2(x)
        x3=self.block3(x)
        x4=self.block4(x)

        x = self.relu(self.conv2(torch.cat((x1,x2,x3,x4),dim=1)))
        x,c=self.lstm(x)
        x=self.tan(x)
        x=self.dense(x)
        x=x.reshape(1,1,256)

        return x,x
