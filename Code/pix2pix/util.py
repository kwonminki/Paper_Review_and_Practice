#전역에 사용되는 것들
import os
import numpy as np

import torch
import torch.nn as nn

from scipy.stats import poisson
from skimage.transform import rescale, resize

from mkdir import mkdirlist


## 네트워크 grad 설정하기
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def save(ckpt_dir, net, optim, epoch, isgan=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if isgan is not None: #net도 dic 안에 list로 줘야함.

        save_dic = {}
        
        for i in range(len(net["generator"])):
            save_dic.update({'netG{}'.format(i) : net["generator"][i].state_dict(), 'optimG{}'.format(i) : optim["generator"][i].state_dict()})
        for j in range(len(net["discriminator"])):
            save_dic.update({'netD{}'.format(j) : net["discriminator"][j].state_dict(), 'optimG{}'.format(j) : optim["discriminator"][j].state_dict()})
        
        torch.save(save_dic, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))
    
    else:
        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                '%s/model_epoch%d.pth' % (ckpt_dir, epoch))


## 네트워크 불러오기
def load(ckpt_dir, net, optim, index = -1, isgan=None):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[index]))
    print(ckpt_lst[index])
   # print(dict_model)
    if isgan is not None:
        for i in range(len(net["generator"])):
            net["generator"][i].load_state_dict(dict_model['netG0'])
            optim["generator"][i].load_state_dict(dict_model['optimG{}'.format(i)])
        for i in range(len(net["discriminator"])):
            net["discriminator"][i].load_state_dict(dict_model['netD{}'.format(i)])
            optim["discriminator"][i].load_state_dict(dict_model['optimD{}'.format(i)]) 
        epoch = int(ckpt_lst[index].split('epoch')[1].split('.pth')[0])


    else:
        net.load_state_dict(dict_model['net'])
        optim.load_state_dict(dict_model['optim'])
        epoch = int(ckpt_lst[index].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


## Add Sampling
##opts로 인자를 넘겨주면 됨.
##1. uniform : opts에는 [y,x] 중심좌표
##2. random : opts에는 [probability] 확률
##3. gaussian : opts에는 [x0, y0, sigma_x, sigma_y, A]  - Wiki
#A is the amplitude, xo,yo is the center 
#and σx, σy are the x and y spreads of the blob. 
#The figure on the right was created using A = 1, xo = 0, yo = 0, σx = σy = 1. 
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(img.shape)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk

    elif type == "random":
        prob = opts[0]

        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < prob).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float)

        dst = img * msk

    elif type == "gaussian":
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(-((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk

    return dst

## Add Noise
##sigma가 크면 노이즈가 심해진다.
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0]

        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])

        dst = img + noise

    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst


"""
----------------------
order options
----------------------
0: Nearest-neighbor
1: Bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic
"""
## Add blurring
## scale은 opts 첫번째 크기로, 다시 되돌리려면 opts 2번째에 1 주면 됨.
def add_blur(img, type="bilinear", opts=None):
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    sz = img.shape
    if len(opts) == 1:
        keepdim = True
    else:
        keepdim = opts[1]

    # dw = 1.0 / opts[0]
    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0] // opts[0], sz[1] // opts[0], sz[2]), order=order)

    if keepdim: #디멘전을 다시 돌려줌.
        # dst = rescale(dst, scale=(1 / dw, 1 / dw, 1), order=order)
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst