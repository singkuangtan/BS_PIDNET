import torch.nn as nn
import torch

class NonNegativeConv2d(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding2=0, bias=True, dilation=1, groups=1):
        super(NonNegativeConv2d, self).__init__()
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size,kernel_size))
        nn.init.kaiming_uniform_(self.weight) #xavier_uniform(self.weight)
        self.weight.data[self.weight.data<0]=0

        bias=True #False #True # set to false to test whether it can work without bias
        #print(self.weight)
        if bias==False:
            self.bias=None
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        if stride<=0:
            stride=1
        #print(stride)
        self.stride = stride
        #print(padding2)
        if padding2==True:
            padding2=1
        self.padding2 = padding2
        #print(self.padding)
        self.kernel_size = kernel_size

        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        #print(self.padding)
        #weight = torch.relu(self.weight)  # Applying ReLU to ensure non-negativity
        weight=self.weight.clone()
        #weight[:,0:weight.shape[1]//2,:,:]=(weight[:,0:weight.shape[1]//2,:,:]+weight[:,weight.shape[1]//2:,:,:])/2
        #weight[:,weight.shape[1]//2:,:,:]=weight[:,0:weight.shape[1]//2,:,:]

        weight.clamp_(0, float('inf'))
        return nn.functional.conv2d(x, weight,stride=[self.stride], padding=[self.padding2, self.padding2], bias=self.bias,dilation=self.dilation,groups=self.groups) #dilation=[1] ,groups=[1],)


class bsconv_layer(nn.Module):

    def __init__(self, inplanes, outplanes,kernel_size, stride=1, padding=0, bias=True, dilation=1, groups=1): #scale_factor=None):
        super(bsconv_layer, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes) #, momentum=bn_mom)

        bias=True # set bias to always true because I think bsconv cannot work without bias

        #self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.conv1=NonNegativeConv2d(2*inplanes, outplanes, kernel_size, stride, padding, bias,dilation,groups)
        #self.relu = nn.ReLU(inplace=True)

        self.weight=self.conv1.weight


    def forward(self, x):

        #x = self.conv1(self.relu(self.bn1(x)))
        #x = self.conv1(torch.cat((self.relu(self.bn1(x)), -self.relu(self.bn1(x))), dim=1))
        x = self.conv1(torch.cat((x, -x) , dim=1))

        return x
