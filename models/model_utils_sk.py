# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from models.bsconv import bsconv_layer as Conv2d
from models.bsconv import NonNegativeConv2d
import torch.nn.functional as F

class CReLU(nn.Module):
    def __init__(self, dim=1, inplace=False):
        super(CReLU, self).__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x):
        # ReLU on x and -x
        pos = F.relu(x, inplace=self.inplace)
        neg = F.relu(-x, inplace=self.inplace)
        return torch.cat([pos, neg], dim=self.dim)

#BatchNorm2d = nn.BatchNorm2d

BatchNorm2d = lambda num_features, **kwargs: nn.GroupNorm(1, num_features) #, **kwargs)

#nn.ReLU=CReLU

bn_mom = 0.1
algc = False

class LinearAddModel(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(LinearAddModel, self).__init__()
        # Define two learnable parameters, initialized to 0.5 for simplicity
        #self.weight1 = nn.Parameter(torch.rand(1))
        #self.weight2 = nn.Parameter(torch.rand(1))
        #self.weight3 = nn.Parameter(torch.rand(1))
        #self.weight4 = nn.Parameter(torch.rand(1))
        #self.weight1.data[self.weight1.data<0]=0
        #self.weight2.data[self.weight2.data<0]=0
        #self.weight3.data[self.weight3.data<0]=0
        #self.weight4.data[self.weight4.data<0]=0
        #self.conv1=build_conv_layer(1, outplanes, 1, stride=1, padding=0, bias=False,groups=inplanes)
        #self.conv2=build_conv_layer(1, outplanes, 1, stride=1, padding=0, bias=False,groups=inplanes)

        self.conv1=NonNegativeConv2d(1, outplanes, 1, stride=1, padding2=0, bias=False,groups=inplanes)
        self.conv2=NonNegativeConv2d(1, outplanes, 1, stride=1, padding2=0, bias=False,groups=inplanes)
        self.conv3=NonNegativeConv2d(1, outplanes, 1, stride=1, padding2=0, bias=False,groups=inplanes)
        self.conv4=NonNegativeConv2d(1, outplanes, 1, stride=1, padding2=0, bias=False,groups=inplanes)
    def forward(self, tensor1, tensor2):

        # Perform the weighted addition of the two tensors
        #weight1=self.weight1.clone()
        #weight1.clamp_(0, float('inf'))
        #weight2=self.weight2.clone()
        #weight2.clamp_(0, float('inf'))
        #weight3=self.weight3.clone()
        #weight3.clamp_(0, float('inf'))
        #weight4=self.weight4.clone()
        #weight4.clamp_(0, float('inf'))
        #weight1=(weight1+weight3)/2
        #weight2=(weight2+weight4)/2
        #weight3=weight1
        #weight4=weight2
        #output = weight1 * tensor1 + weight2 * tensor2+weight3 * -tensor1 + weight4 * -tensor2
        #output=self.conv1(tensor1)+self.conv2(tensor2)
        #print(tensor1.shape,tensor2.shape)
        output=self.conv1(tensor1)+self.conv3(-tensor1)+self.conv2(tensor2)+self.conv4(-tensor2)
        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        self.lam=LinearAddModel(planes,planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out=self.lam(out,residual)
        #out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        self.lam=LinearAddModel(planes* self.expansion,planes* self.expansion)

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

        out=self.lam(out,residual)
        #out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=BatchNorm2d): #nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

        self.lam=LinearAddModel(outplanes,outplanes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out=self.lam(self.compression(torch.cat(x_list, 1)),self.shortcut(x))
        #out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    
class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=BatchNorm2d): #nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

        self.lam=LinearAddModel(outplanes,outplanes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
       
        out=self.lam(self.compression(torch.cat([x_,scale_out], 1)),self.shortcut(x))
        #out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out
    

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=BatchNorm2d): #nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
        self.lam=LinearAddModel(mid_channels,mid_channels)

        #if self.with_channel==True:
        self.lam2=LinearAddModel(in_channels,in_channels)
        self.lam3=LinearAddModel(in_channels,in_channels)

        self.lam4=LinearAddModel(in_channels,in_channels)
        #else:
        #    self.lam2=LinearAddModel(1,1)
        #    self.lam3=LinearAddModel(1,1)

        #    self.lam4=LinearAddModel(1,1)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            #sim_map = torch.sigmoid(self.up(x_k * y_q))
            sim_map = torch.relu(self.up(self.lam(x_k , y_q)))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
            #sim_map = torch.relu(torch.sum(self.lam(x_k , y_q), dim=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        #x = (1-sim_map)*x + sim_map*y
        sim_map=sim_map.repeat(1,input_size[2],1,1)
        x = self.lam4(self.lam2((1-sim_map),x) , self.lam3(sim_map,y))

        return x
    
class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=BatchNorm2d): #nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
                                Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        
        self.lam=LinearAddModel(in_channels,in_channels)
        self.lam2=LinearAddModel(in_channels,in_channels)

        self.lam3=LinearAddModel(in_channels,in_channels)
        self.lam4=LinearAddModel(in_channels,in_channels)

        self.lam5=LinearAddModel(out_channels,out_channels)

    def forward(self, p, i, d):
        #edge_att = torch.sigmoid(d)
        edge_att = torch.relu(d)
        
        #p_add = self.conv_p((1-edge_att)*i + p)
        #i_add = self.conv_i(i + edge_att*p)
        p_add = self.conv_p(self.lam3(self.lam((1-edge_att),i) , p))
        i_add = self.conv_i(self.lam4(i , self.lam2(edge_att,p)))

        #return p_add + i_add
        return self.lam5(p_add , i_add)
    

class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=BatchNorm2d): #nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        
        self.lam=LinearAddModel(in_channels,in_channels)
        self.lam2=LinearAddModel(in_channels,in_channels)

        self.lam3=LinearAddModel(in_channels,in_channels)
        self.lam4=LinearAddModel(in_channels,in_channels)

        self.lam5=LinearAddModel(out_channels,out_channels)

    def forward(self, p, i, d):
        #edge_att = torch.sigmoid(d)
        edge_att = torch.relu(d)
        
        #p_add = self.conv_p((1-edge_att)*i + p)
        #i_add = self.conv_i(i + edge_att*p)
        
        #return p_add + i_add

        p_add = self.conv_p(self.lam3(self.lam(1-edge_att),i) , p)
        i_add = self.conv_i(self.lam4(i , self.lam2(edge_att,p)))
        
        return self.lam5(p_add , i_add)

class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=BatchNorm2d): #nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                Conv2d(in_channels, out_channels, 
                                          kernel_size=3, padding=1, bias=False)                  
                                )

        self.lam=LinearAddModel(in_channels,in_channels)
        self.lam2=LinearAddModel(in_channels,in_channels)
        self.lam3=LinearAddModel(in_channels,in_channels)

    def forward(self, p, i, d):
        #edge_att = torch.sigmoid(d)
        #return self.conv(edge_att*p + (1-edge_att)*i)

        edge_att = torch.relu(d)
        return self.conv(self.lam3(self.lam(edge_att,p) , self.lam2((1-edge_att),i)))
    


if __name__ == '__main__':

    
    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()
    
    out = net(x,y)