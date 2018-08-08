from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output= output.mean(0)
        return output.view(1)
            
###########################################################################
#  MLP for 8 Gaussian, hidden_node_number_Ref: github_poolio_unrolledGAN  #
###########################################################################

class Gauss_G(nn.Module):
    def __init__(self, nz=256, ngf=128, isize=2, ngpu=1):
        super(Gauss_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ngf, ngf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ngf, ngf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ngf, isize, bias=True),
        )
        self.main = main
        self.nz = nz
        self.isize = isize

    def forward(self, input):
        input = input.view(input.size(0), -1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), -1)


class Gauss_D(nn.Module):
    def __init__(self, isize=2, ndf=128, ngpu=1):
        super(Gauss_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(isize, ndf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ndf, ndf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ndf, ndf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ndf, 1, bias=True),
        )
        self.main = main

    def forward(self, input):
        input = input.view(input.size(0), -1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1)

###############################################
# For reconstruction, input is 1 always, z is treated as weight
###############################################        
class REC(nn.Module):
    def __init__(self, nz):
        super(REC, self).__init__()

        main = nn.Sequential(
            nn.Linear(1, nz, bias=False),
        )
        self.main = main
        self.nz = nz

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(intput.size(0), nz, 1, 1)
