from typing import Union
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
import numpy as np
import torch.nn.functional as F
import math

class Conv2d(_ConvNd):
   
    def __init__(
        self,
        in_channels: int, #输入图像的通道数或者说深度
        out_channels: int, #输出图像的通道数或者说深度，决定了卷积核的个数
        kernel_size: _size_2_t, #卷积核尺寸（宽和高）
        stride: _size_2_t = 1,  #步长
        padding: Union[str, _size_2_t] = 0, #边缘拓展的数目
        dilation: _size_2_t = 1, #类似与卷积的稠密程度，若为1，则就是最标准的卷积，若为2，则会变成类似空洞卷积的情况
        groups: int = 1,  #是个控制输入输出通道分组的情况，一般默认为1，若为2，则输入和输出的通道数要对半分组
        bias: bool = True,  #偏置
        padding_mode: str = 'zeros',  #padding的格式 # TODO: refine this type
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        
    def conv2d(self, input:Tensor, kernel:Tensor, bias = 0, stride=1, padding=0):
         if padding > 0:
           input = F.pad(input, (padding, padding, padding, padding))
         bs,in_channels,input_h, input_w = input.shape
         out_channel, in_channel,kernel_h, kernel_w = kernel.shape
         #input = input.view(input.size(0), -1)
         #kernel = kernel.view(kernel.size(0), -1)
         output_h = (math.floor((input_h - kernel_h) / stride) + 1)
         output_w = (math.floor((input_w - kernel_w) / stride) + 1)

         if bias is None:
            bias = torch.zeros(out_channel)

    # 初始化输出矩阵
         output = torch.zeros(bs, out_channel, output_h, output_w)
         
         for ind in range(bs): #控制batch-size
          for oc in range(out_channel):   #
            for ic in range(in_channel):  #这两层是通过计算出的输出矩阵进行卷积核运动的逻辑控制
                for i in range(0, input_h - kernel_h + 1, stride): #对运动进行具体控制
                    for j in range(0, input_w - kernel_w + 1, stride):
                        region = input[ind, ic, i:i + kernel_h, j: j + kernel_w]
                        # 点乘相加
                        output[ind, oc, int(i / stride), int(j / stride)] += torch.sum(region * kernel[oc, ic])
            output[ind, oc] += bias[oc]


         return output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)
    
    def backward(self, ones: Tensor):
        '''TODO backward的计算方法''' 
        return self.input.grad
    
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))#随机weight
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            
    def forward(self, input): #输入的维数是否可控？
           self.input=input
           self.output = torch.addmm(self.bias, input, self. weight.t()) #.t()将tensor进行转置，这行起到将后两个参数进行内积再加上bias偏置。
        
        
           return self.output
    def backward(self, ones: Tensor):#
        #self.weight.grad= torch.mm(self.input.T,ones)
        self.bias.grad=ones
        #self.input.grad =   
        return torch.mm(ones,self.weight.T)  #self.input.grad
       
        

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        self.output = 0.
        for i in range(input.shape[0]):

            numerator = torch.exp(input[i, target[i]])     # 分子
            denominator = torch.sum(torch.exp(input[i, :]))   # 分母

            # 计算单个损失
            loss = -torch.log(numerator / denominator)
           
            #print("单个损失： ",loss)

            # 损失累加
            self.output += loss

        # 整个 batch 的总损失是否要求平均
        
        self.output /= input.shape[0]

        
        return self.output
    def backward(self):
        '''TODO'''
        return self.input.grad
        
