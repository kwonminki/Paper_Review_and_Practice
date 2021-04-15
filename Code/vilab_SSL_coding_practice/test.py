import argparse

import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

from model import *
from dataset import * #모두다
from util import * #save, load
from mkdir import mkdirlist


def test(args):
        #하이퍼파라미터
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    norm = args.norm

    #디렉토리
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    #args
    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    train_continue = args.train_continue
    network = args.network
    learning_type = args.learning_type

    device = args.device
    mode = args.mode

    cmap = None #"gray" 하면 흑백으로 저장됨
    isgan = "yes" # GAN인지
    SAVE_NUM = 3 # epoch 몇번마다 save 할지

    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 재설정.
    ckpt_dir = ckpt_dir/task/network/learning_type
    log_dir = log_dir/task/network/learning_type
    result_dir = result_dir/task/network/learning_type

    result_dir_test = result_dir/"test"

    #결과 저장용 디렉토리 생성
    mkdirlist([data_dir,ckpt_dir,log_dir,result_dir, result_dir_test])
    if not os.path.exists(result_dir_test/'png'):
        os.makedirs(result_dir_test/'png')
        os.makedirs(result_dir_test/'numpy')




    if mode == 'test':
        #테스트 데이터
        transform_test = transforms.Compose([Resize(shape=(ny,nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_test = Dataset(data_dir, transform=transform_test, task=task, opts=opts)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
        
        #필요해서 만듬
        num_data_test = len(dataset_test)

        #batch 사이즈
        num_batch_test = np.ceil(num_data_test/batch_size)

    ##
    #2
    ##네트워크 생성 -> GAN 용으로 바꿈. 나중엔 generator, discriminator도 여러개가 될 수 있으므로 그에 맞게 만듬.
    if network == "dcgan":
        netG = DCGAN_Generator(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = DCGAN_Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        net = {"generator" : [netG], "discriminator" : [netD]}


    #loss는 MSE로
    fn_loss = nn.BCELoss().to(device)
    #fn_loss = nn.BCEWithLogitsLoss().to(device)

    #Adam사용
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999)) #논문에서 변경함.
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optim = {"generator" : [optimG], "discriminator" : [optimD]}

    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean
    #fn_binaryclass = lambda x: 1.0*(x>0.5)


    if mode == "test":
        ## 네트워크 학습시키기
        st_epoch = 0
        net, optim, st_epoch = load(ckpt_dir, net, optim, isgan=isgan)

        with torch.no_grad():
            net["generator"][0].eval()
            input = torch.randn(batch_size, 100, 1, 1).to(device)
            output = net["generator"][0](input)

            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(output.shape[0]):
                id = j

                output_ = output[j]
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)
                output_ = np.clip(output_, a_min=0, a_max=1)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_.squeeze(), cmap=cmap)

