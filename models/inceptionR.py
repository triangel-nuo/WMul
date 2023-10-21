import logging
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['InceptionR']


model_urls = {
    'inceptionR': './models/inceptionR.pth',
}

##########################################coordinate attention#############################################
def swish(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

class CA(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=16):
        super(CA, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = swish(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h
##############################################coordinate attention############################################    
    
class SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

    
##############################################CBAM attention############################################  
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #-----------------------------------------#
        # 利用1x1卷积代替全连接,以减小计算量以及模型参数
        # 整体前向传播过程为 卷积 ReLU 卷积 进而实现特征
        # 信息编码以及增强网络的非线性表达能力
        #-----------------------------------------#
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        #----------------------------------------#
        # 定义sigmoid激活函数以增强网络的非线性表达能力
        #----------------------------------------#
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        #-------------------------------------#
        # 分成两部分,一部分为平均池化,一部分为最大池化
        # 之后将两部分的结果相加再经过sigmoid作用
        #-------------------------------------#
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out     = avg_out + max_out
        return self.sigmoid(out)
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding      = 3 if kernel_size == 7 else 1
        #-------------------------------------#
        # 这个卷积操作为大核卷积操作,其虽然可以计算
        # 空间注意力但是仍无法有效建模远距离依赖关系
        #-------------------------------------#
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        #----------------------------------------------------#
        # 整体前向传播过程即为先分别做平均池化操作,再做最大池化操作
        # 其中的dim：
        # 指定为1时,求得是列的平均值
        # 指定为0时,求得是行的平均值
        # 之后将两个输出按照列维度进行拼接,此时通道数为2
        # 拼接之后通过一个大核卷积将双层特征图转为单层特征图,此时通道为1
        # 最后通过sigmoid来增强模型的非线性表达能力
        #-----------------------------------------------------#
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x          = torch.cat([avg_out, max_out], dim=1)
        x          = self.conv1(x)
        return self.sigmoid(x)

    
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        #----------------------------#
        # 定义好通道注意力以及空间注意力
        #----------------------------#
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
 
        #----------------------------#
        # 输入x先与通道注意力权重相乘
        # 之后将输出与空间注意力权重相乘
        #----------------------------#
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
##############################################CBAM模块############################################ 

##############################################ECA模块############################################ 
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        #----------------------------------#
        # 根据通道数求出卷积核的大小kernel_size
        #----------------------------------#
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()
 
    def forward(self, x):
        #------------------------------------------#
        # 显示全局平均池化,再是k*k的卷积,
        # 最后为Sigmoid激活函数,进而得到每个通道的权重w
        # 最后进行回承操作,得出最终结果
        #------------------------------------------#
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
##############################################ECA模块############################################ 

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    
class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
    
    
class Block2(nn.Module):

    def __init__(self, scale=1.0):
        super(Block2, self).__init__()

        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.act(out)
        return out

    
    
class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block3(nn.Module):

    def __init__(self,  scale=1.0):
        super(Block3, self).__init__()
        
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.act(out)
        return out


    
class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block4(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block4, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.act(out)
        return out

    
class InceptionR(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionR, self).__init__()

        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        
        self.repeat = nn.Sequential(
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17),
            Block2(scale=0.17)
        )
        
        self.mixed_6a = Mixed_6a()
        
        self.repeat2 = nn.Sequential(
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10),
            Block3(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        
        self.repeat3 = nn.Sequential(
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20),
            Block4(scale=0.20)
        )
        
        self.block4 = Block4(noReLU=True)
        
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)
        
        # Drop系列
        # self.stochastic_depth = torchvision.ops.StochasticDepth(0.2,"row")
        # self.dropblock = torchvision.ops.DropBlock2d(0.2, 7, inplace = True)
        # self.dropout = nn.Dropout(p=0.2)
        
        self.features = nn.Sequential(
            #Block1
            self.conv2d_1a,
            torchvision.ops.DropBlock2d(0.008, 7),
            self.conv2d_2a,
            torchvision.ops.DropBlock2d(0.015, 7),
            self.conv2d_2b,
            self.maxpool_3a,
            torchvision.ops.DropBlock2d(0.023, 7),
            self.conv2d_3b,
            torchvision.ops.DropBlock2d(0.030, 7),
            self.conv2d_4a,
            self.maxpool_5a,
            torchvision.ops.DropBlock2d(0.038, 7),
            self.mixed_5b,
            torchvision.ops.DropBlock2d(0.046, 7),

            self.repeat,
            torchvision.ops.DropBlock2d(0.053, 7),
            self.mixed_6a,
            torchvision.ops.DropBlock2d(0.061, 7),

            self.repeat2,
            torchvision.ops.DropBlock2d(0.068, 7),
            self.mixed_7a,
            torchvision.ops.DropBlock2d(0.076, 7),

            self.repeat3,
            torchvision.ops.DropBlock2d(0.084, 7),
            self.block4,
            torchvision.ops.DropBlock2d(0.091, 7),
            self.conv2d_7b,
            torchvision.ops.DropBlock2d(0.098, 7),
        )

    
    def logits(self, features):
        x = self.avgpool_1a(features)
        # x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    
    def get_features(self):
        return self.features

    
    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(InceptionR, self).load_state_dict(model_dict)
        

        
def inceptionR(num_classes=1000, pretrained=False):
    model = InceptionR(num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load(model_urls['inceptionR']))
        new_last_linear = nn.Linear(1536, 1000)
        new_last_linear.weight.data = model.last_linear.weight.data[1:]
        new_last_linear.bias.data = model.last_linear.bias.data[1:]
        model.last_linear = new_last_linear
    return model
        
