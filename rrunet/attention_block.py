import paddle
import paddle.nn as nn
import os
import numpy as np
import cv2
from paddle import fluid
from paddle.nn import initializer

# from rrunet.unet_model import *


'''
att: [1, 256, 16, 16]
att: [1, 256, 32, 32]
att: [1, 128, 64, 64]
att: [1, 64, 128, 128]
att: [1, 32, 256, 256]
'''
# 113
class SelfAttention(nn.Layer):
	def __init__(self,in_channel,r=16):
		super(SelfAttention,self).__init__()
		self.linear1=nn.Sequential(nn.Conv2D(in_channel,in_channel//r,1,1))
		self.linear2=nn.Sequential(nn.Conv2D(in_channel,in_channel//r,1,1))
		self.relation=nn.Sequential()
		self.trans=nn.Sequential(nn.Conv2D(in_channel//r,in_channel//r,1,1),nn.ReLU(),nn.Conv2D(in_channel//r,in_channel//r,1,1))
		self.agg=nn.Sequential(nn.BatchNorm2D(in_channel//r),nn.ReLU(),nn.Conv2D(in_channel//r,in_channel,1,1))

	def forward(self,inputx):
		x1=self.linear1(inputx)
		x2=self.linear2(inputx)
		x2=self.relation(x2)
		x2=self.trans(x2)
		x=x1*x2
		return inputx+self.agg(x)

print(paddle.__version__)
SA=SelfAttention(256)
x=paddle.randn([1,256,16,16])
y=SA(x)
# print(y)

# 将h和w进行打平到一个维度
def hw_flatten(x):
    b, c, h, w = x.shape
    x =  fluid.layers.reshape(x, shape=(b, c, h*w))
    return x

class Attention(nn.Layer):
	def __init__(self,in_c,channel,down=False):
		super(Attention, self).__init__()
		self.channel = channel
		self.down = down
		self.grid = paddle.nn.Conv2D(in_c,in_c,16,16)
		self.f = paddle.nn.Conv2D(
			in_channels=in_c, out_channels=channel // 8,
			kernel_size=1, stride=1,
			padding='SAME', data_format='NCHW')

		self.g = paddle.nn.Conv2D(
			in_channels=in_c, out_channels=channel // 8,
			kernel_size=1, stride=1,
			padding='SAME', data_format='NCHW')

		self.h = paddle.nn.Conv2D(
			in_channels=in_c, out_channels=channel,
			kernel_size=1, stride=1,
			padding='SAME', data_format='NCHW')

		self.o = paddle.nn.Conv2D(
			in_channels=channel, out_channels=channel,
			kernel_size=1, stride=1,
			padding='SAME', data_format='NCHW')

		self.up = nn.Sequential(
			# nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
			nn.Conv2DTranspose(channel, channel, 16, stride=16),
			nn.GroupNorm(32, channel))

		self.gamma = paddle.zeros([1])
		#paddle.fluid.layers.create_parameter(shape=[1], name='gamma', dtype='float32',default_initializer=fluid.initializer.ConstantInitializer(value=0.0))


	def forward(self, x):
		n_, c_, h_, w_ = x.shape
		# print("att:",x.shape)
		if self.down:
			grx=self.grid(x)
			_,_,h_,w_=grx.shape
		else:
			grx = x
		fx=self.f(grx)  # b,32,h,w
		gx=self.g(grx)  # b,4,h,w
		hx=self.h(grx)  # b,32,h,w
		s = fluid.layers.matmul(hw_flatten(fx), hw_flatten(gx), transpose_x=True)  # b,h*w,h*w
		# print(x.shape,fx.shape,gx.shape,hx.shape,hw_flatten(fx).shape,s.shape)
		attention_ = fluid.layers.softmax(s)  # b,h*w,h*w

		o = fluid.layers.matmul(hw_flatten(hx), attention_)  # b c n
		o = fluid.layers.reshape(o, shape=(n_, self.channel, h_, w_))  # b c h w
		o = self.o(o)
		if self.down:
			x = self.gamma * self.up(o) + x
		else:
			x = self.gamma * o + x

		return x

class HeadsAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # 计算 q,k,v 的转移矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 最终的线性层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape[1:]
        # 线性变换
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        # 分割 query key value
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scaled Dot-Product Attention
        # Matmul + Scale
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        # SoftMax
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # Matmul
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        # 线性变换
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# x=paddle.randn([1,1024,16,16])
# y=Attention(1024,1024)(x)
# print(x.flatten(2).shape)#.transpose((0, 2, 1))
# y=HeadsAttention(256)(x.flatten(2))#.transpose((0, 2, 1))
# print(y.shape)


'''
att: [1, 256, 16, 16]
att: [1, 256, 32, 32]
att: [1, 128, 64, 64]
att: [1, 64, 128, 128]
att: [1, 32, 256, 256]

# print("shape",x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
# shape[16, 32, 256, 256][16, 64, 128, 128][16, 128, 64, 64][16, 256, 32, 32][16, 256, 16, 16]
'''

class AdaFusion(nn.Layer):

	def __init__(self):
		super(AdaFusion, self).__init__()
		self.a = paddle.to_tensor([1], stop_gradient=False)*0.3
		self.b = paddle.ones([1])*0.3
		self.c = paddle.ones([1])*0.3

	def forward(self, x1, x2, x3):
		return self.a*x1+self.b*x2+self.c*x3

class AdaFusionP(nn.Layer):

	def __init__(self, size):
		super(AdaFusionP, self).__init__()
		self.weight = 1.0/size
		self.layers = [paddle.ones([1])*self.weight for _ in range(size)]

	def forward(self, *inputs):
		x = 0
		for i in range(len(inputs)):
			x += self.layers[i] * inputs[i]
		return x


class AdaFusionR(nn.Layer):

	def __init__(self, size):
		super(AdaFusionR, self).__init__()
		self.size = size
		self.weight = 1.0/size
		self.layers = [paddle.ones([1])*self.weight for _ in range(size)]
		# self.layers = [paddle.create_parameter([1], dtype="float32", default_initializer=initializer.Constant(self.weight))  for _ in range(size)]
		self.a = paddle.create_parameter([1], dtype="float32", default_initializer=initializer.Constant(self.weight))
		self.b = paddle.create_parameter([1], dtype="float32", default_initializer=initializer.Constant(self.weight))
		self.c = paddle.create_parameter([1], dtype="float32", default_initializer=initializer.Constant(self.weight))
		# self.layers = nn.Sequential(*self.layers)

	def forward(self, *inputs):
		# x = 0
		# for i in range(len(inputs)):
		# 	x += self.layers[i] * inputs[i]
		# return x

		if len(inputs)==2:
			return self.a * inputs[0] + self.b * inputs[1]
		else:
			return self.a * inputs[0] + self.b * inputs[1] + self.c * inputs[2]

class TestModle(nn.Layer):
	def __init__(self):
		super(TestModle, self).__init__()
		self.ada = AdaFusionR(3)

	def forward(self, *inputs, **kwargs):
		return self.ada(inputs[0],inputs[0],inputs[0])

# flops = paddle.flops(TestModle(),[1,3,128,128],print_detail=True)

class AdaFusion2(nn.Layer):

	def __init__(self):
		super(AdaFusion2, self).__init__()
		self.a = paddle.ones([1])*0.5
		self.b = paddle.ones([1])*0.5

	def forward(self, x1, x2):
		return self.a * x1 + self.b * x2

#
# x1=paddle.randn([1,1024,16,16])
# x2=paddle.randn([1,1024,16,16])
# fusion=AdaFusion2()
# y=fusion(x1,x2)
# print(y.shape)
# # print(y)
#
# fusion=AdaFusionP(2)
# y=fusion(x1,x2)
# print(y.shape)
# print(y)

class FusionModule(nn.Layer):

	def __init__(self, in_ch, num=3):
		super(FusionModule, self).__init__()
		self.fusion1 = AdaFusionP(num)
		self.fusion2 = AdaFusionP(num)
		self.conv1_1 = nn.Conv2D(in_ch // 2, in_ch // 2, kernel_size=3, dilation=1, padding="SAME")
		self.conv1_2 = nn.Conv2D(in_ch // 2, in_ch // 2, kernel_size=3, dilation=3, padding="SAME")
		self.conv1_3 = nn.Conv2D(in_ch // 2, in_ch // 2, kernel_size=3, dilation=5, padding="SAME")
		self.conv2_1 = nn.Conv2D(in_ch // 2, in_ch // 2, kernel_size=3, dilation=1, padding="SAME")
		self.conv2_2 = nn.Conv2D(in_ch // 2, in_ch // 2, kernel_size=3, dilation=3, padding="SAME")
		self.conv2_3 = nn.Conv2D(in_ch // 2, in_ch // 2, kernel_size=3, dilation=5, padding="SAME")

	def forward(self, *inputs, **kwargs):
		x1 = inputs[0]
		x2 = inputs[1]

		x1 = self.fusion1(self.conv1_1(x1), self.conv1_2(x1), self.conv1_3(x1))
		x2 = self.fusion2(self.conv2_1(x2), self.conv2_2(x2), self.conv2_3(x2))

		return x1, x2