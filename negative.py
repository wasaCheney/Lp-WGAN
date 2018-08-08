# Cheney - Mon Sept 18 2017
# PyTorch implementation for Lp-WGAN
# Reference: WGAN

from __future__ import print_function
import argparse
import random
import os

import gc
import resource

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable


from gradient_penalty import calc_gradient_penalty
from dcgan import dcgan
from mlp import mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--seed', type=int, default=1123)
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--norm', required=True, help='l1 | l2 | linfty | vanilla')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--gradient_penalty', action='store_true', help='Whether to use gradient penalty')
opt = parser.parse_args()
print(opt)

interval = 500 # storage interval
if opt.experiment is None:
    opt.experiment = '{}_negative_before_{}_lr_{}_cu_{}'.format(opt.dataset, opt.norm, opt.lrG, opt.clamp_upper)
os.system('mkdir {0}'.format(opt.experiment))

random.seed(opt.seed)
torch.manual_seed(opt.seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'fashionmnist': # very similar to MNIST
    dataset = dset.FashionMNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1) # XXX opt.batchSize

#alpha = torch.FloatTensor(opt.batchSize, 1)
#interpolates = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    #noise, fixed_noise, alpha, interpolates = noise.cuda(), fixed_noise.cuda(), alpha.cuda(), interpolates.cuda()
    #fixed_noise = fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD) # adaptive learning rate SGD for mini-batch optimization
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)
    

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1
            
            if (opt.norm == 'vanilla') and (not opt.gradient_penalty): # vanilla wgan
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            elif (not opt.gradient_penalty): # XXX
                for name, p in netD.named_parameters():
                    #print(name.split('.')[-2].split('_')[-1]) # XXX
                    batchnorm = True if name.split('.')[-2].split('_')[-1] == 'batchnorm' else False # TODO
                    if batchnorm:
                        upperb = np.percentile(p.data.abs().cpu().numpy(), 95)
                        p.data.clamp_(-float(upperb), float(upperb))
                        if opt.norm == 'linfty':
                            normb = upperb
                        elif opt.norm == 'l2':
                            normb = p.data.norm(2)
                        elif opt.norm == 'l1':
                            normb = p.data.norm(1)
                        else:
                            assert normb, 'NormError: Only vanilla, l1, l2 and linfty are accepted!'
                        p.data.mul_(opt.clamp_upper).div_(normb + 1e-8)
                        continue
                    conv = True if name.split('.')[-2].split('_')[-1] == 'conv' else False
                    if conv:
                        upperc = np.percentile(p.data.view(p.data.size(0), -1).abs().cpu().numpy(), 95, axis=1) # 95th percentile rather than maximum
                        for ind, ele in enumerate(upperc):
                            p.data[ind].clamp_(-ele, ele)
                        if opt.norm == 'linfty':
                            normc = upperc
                        elif opt.norm == 'l2':
                            normc = p.data.view(p.data.size(0), -1).norm(2, 1)
                        elif opt.norm == 'l1':
                            normc = p.data.view(p.data.size(0), -1).norm(1, 1)
                        else:
                            assert normc, 'NormError: Only vanilla, l1, l2 and linfty are accepted!'
                        for ind, ele in enumerate(normc):
                            p.data[ind].mul_(opt.clamp_upper).div_(ele + 1e-8)

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            #input = real_cpu
            realv = Variable(input)

            errD_real = netD(realv).mean()
            errD_real.backward(mone)

            # train with fake
            #noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            with torch.no_grad():
                noisev = Variable(noise) # totally freeze netG
                fake = netG(noisev)
            inputv = Variable(fake.data)
            errD_fake = netD(inputv).mean()
            errD_fake.backward(one)
            errD = errD_real - errD_fake # XXX
            
            # train with gradient penalty
            if opt.gradient_penalty:
                gradient_penalty = calc_gradient_penalty(netD, realv.data, fake.data, opt.batchSize, LAMBDA=10, use_cuda=True) # XXX
                #gradient_penalty = calc_gradient_penalty(netD, realv.data, fake.data)
                gradient_penalty.backward()
                '''
                torch.rand(alpha.size(), out=alpha) # uniform
                interpolates.copy_(alpha.view(-1, 1, 1, 1) * realv.data + ((1 - alpha.view(-1, 1, 1, 1)) * fake.data))
                interpolatesv = Variable(interpolates, requires_grad=True)
                
                disc_interpolatesv = netD(interpolatesv)
                gradients = autograd.grad(outputs=disc_interpolatesv, inputs=interpolatesv,
                              grad_outputs=torch.ones(disc_interpolatesv.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolatesv.size()),
                              only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10 # XXX
                
                gradient_penalty.backward()
                '''
                
                errD = errD_real - errD_fake - gradient_penalty # XXX
            #torch.autograd.grad()
            
            #gc.collect()
            #max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #print("{:.2f} MB".format(max_mem_used / 1024))
            
            optimizerD.step()
            netD.zero_grad()
            #del gradient_penalty.grad
            #del gradient_penalty


        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        #noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake).mean()
        errG.backward(mone)
        optimizerG.step()
        netG.zero_grad()
        
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        if gen_iterations % interval == 0:
            #real_cpu = real_cpu.mul(0.5).add(0.5)
            #vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            with torch.no_grad():
                fake = netG(Variable(fixed_noise))
                testErrG = netD(fake).mean()
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
            with open('{0}/loss'.format(opt.experiment), 'a') as f:
                f.write('{} {} {}\n'.format(errD.data[0], errG.data[0], testErrG.data[0]))
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.ckpt'.format(opt.experiment, epoch))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.ckpt'.format(opt.experiment, epoch))
