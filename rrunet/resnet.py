import paddle
import paddle.nn as nn
from paddle.vision.models import resnet50,resnet34

class ResNetImageNet(nn.Layer):
    def __init__(self):
        super(ResNetImageNet,self).__init__()
        self.res = nn.LayerList(resnet50(True).sublayers()[:-22])
    
    def forward(self,x):
        for layer in self.res:
            x=layer(x)
        return x

class Resblock(nn.Layer):

    def __init__(self,in_c,out_c,pool=False):
        super(Resblock,self).__init__()
        self.in_c,self.out_c=in_c,out_c
        self.pool = pool
        if pool==True:
            self.conv=nn.Sequential(nn.Conv2D(in_c,out_c,3,stride=2,padding=1),nn.Conv2D(out_c,out_c,3,padding=1),nn.BatchNorm(out_c))
            self.w=nn.Conv2D(in_c,out_c,kernel_size=1,stride=2)
        else:
            self.conv=nn.Sequential(nn.Conv2D(in_c,out_c,3,padding=1),nn.Conv2D(out_c,out_c,3,padding=1),nn.BatchNorm(out_c))
            self.w=nn.Conv2D(in_c,out_c,kernel_size=1)
    
    def forward(self,x):
        copy_x=x
        x=self.conv(x)
        # if self.in_c!=self.out_c:
        copy_x=self.w(copy_x)
        return x+copy_x
        
class ResNet(nn.Layer):
    
    def __init__(self):
        super().__init__()
        self.layers=nn.LayerList()
        #input:256*256
        self.layers.append(nn.Sequential(nn.Conv2D(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)))#,nn.MaxPool2D(2)
        #->128*128
        for ind in range(3):
            self.layers.append(Resblock(64,64))
        self.layers.append(Resblock(64,128,True))#->64*64
        for ind in range(3):
            self.layers.append(Resblock(128,128))
        self.layers.append(Resblock(128,256,True))#->32*32
        for ind in range(5):
            self.layers.append(Resblock(256,256))
        self.layers.append(Resblock(256,256,True))#->16*16
        for ind in range(2):
            self.layers.append(Resblock(256,256))
        # self.layers.append(nn.Sequential(nn.Flatten(),nn.Linear(100352,2)))
    
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x