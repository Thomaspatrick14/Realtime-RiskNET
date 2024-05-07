import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

from functools import partialmethod

def get_inplanes():
    return [128, 256, 512, 1024]

# def conv3x3x3(in_planes, out_planes, stride=1):
#     return nn.Conv3d(in_planes,
#                      out_planes,
#                      kernel_size=3,
#                      stride=stride,
#                      padding=1,
#                      groups=32,
#                      bias=False)


# def conv1x1x1(in_planes, out_planes, stride=1):
#     return nn.Conv3d(in_planes,
#                      out_planes,
#                      kernel_size=1,
#                      stride=stride,
#                      bias=False)

class ModifiedBasicBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, cardinality, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__()

        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 input_depth,
                 shortcut_type='B',
                 cardinality=32):
        self.inplanes = 64
        super().__init__()
        # block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.conv1 = nn.Conv3d(input_depth,
                               self.inplanes,
                               kernel_size=(7),
                               stride=(1, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, cardinality)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       cardinality,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       cardinality,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       cardinality,
                                       stride=2)
        
        self.avgpool = nn.AvgPool3d(
            (1, 4, 4), stride=1)

        # self.avgpool = nn.AdaptiveAvgPool3d(( 1, 1, 1)) # TODO : Compare with avgpool, which was what TIM had.

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = Variable(torch.cat([out.data, zero_pads], dim=1))

        return out

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    self.downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # print(f"the n_output features: {x.size()}")

        return x

def resnext18(**kwargs):
    """Constructs a modified bottleneck ResNeXt-18 model.
    """
    model = ResNeXt(ModifiedBasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    
    return model

def resnext24(**kwargs):
    """Constructs a modified bottleneck ResNeXt-18 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [2, 2, 2, 2], get_inplanes(), **kwargs)
    
    return model

def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    return model