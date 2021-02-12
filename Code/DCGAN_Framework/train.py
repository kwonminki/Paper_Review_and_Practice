#%%
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

def train(args):
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
    SAVE_NUM = 1 # epoch 몇번마다 save 할지

    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 재설정.
    ckpt_dir = ckpt_dir/task/network/learning_type
    log_dir = log_dir/task/network/learning_type
    result_dir = result_dir/task/network/learning_type

    result_dir_train = result_dir/"train"

    #결과 저장용 디렉토리 생성
    mkdirlist([data_dir,ckpt_dir,log_dir,result_dir, result_dir_train])
    if not os.path.exists(result_dir_train/'png'):
        os.makedirs(result_dir_train/'png')


    if mode == 'train':
        #학습데이터
        transform_train = transforms.Compose([Resize(shape=(ny,nx, nch)), Normalization(mean=0.5, std=0.5)]) #Randomcrop으로 이미지 사이즈 맞춰줌
    
        dataset_train = Dataset(data_dir, transform=transform_train, task=task, opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        #필요해서 만듬
        num_data_train = len(dataset_train)

        #batch 사이즈 
        num_batch_train = np.ceil(num_data_train/batch_size)

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

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


    if mode == "train":
        ## 네트워크 학습시키기
        st_epoch = 0
        #저장해둔 네트워크 불러오기.
        if train_continue == 'on':
            net, optim, st_epoch = load(ckpt_dir, net, optim, isgan=isgan)

        for epoch in range(st_epoch+1, epochs+1):
            if isgan is not None:
                loss_D = {}
                loss_G = {}
                loss_G_train = []
                loss_D_real_train = []
                loss_D_fake_train = []
                for i in range(len(net["generator"])):
                    net["generator"][i].train()
                    loss_G.update({'loss_G{}'.format(i) : []})
                for i in range(len(net["discriminator"])):
                    net["discriminator"][i].train()
                    loss_D.update({'loss_D_real{}'.format(i) : []})
                    loss_D.update({'loss_D_fake{}'.format(i) : []})
            else:
                loss_arr = []

            for batch, data in enumerate(loader_train, 1):
                label = data['label'].to(device)
                
                if isgan is not None:
                    input = torch.randn(label.shape[0], 100, 1, 1).to(device)

                else:
                    input = data['input'].to(device)

                output = net["generator"][0](input)

                #backward netD
                set_requires_grad(net["discriminator"], True)
                optim["discriminator"][0].zero_grad()

                pred_real = net["discriminator"][0](label)
                pred_fake = net["discriminator"][0](output.detach())  #연결 끊음. - detach

                #loss..
                for i in range(len(net["discriminator"])):
                    loss_D['loss_D_real{}'.format(i)] = fn_loss(pred_real, torch.ones_like(pred_real))
                    loss_D['loss_D_fake{}'.format(i)] = fn_loss(pred_fake, torch.zeros_like(pred_fake))
                    loss_D['loss_D_sum{}'.format(i)] = 0.5 * (loss_D['loss_D_real{}'.format(i)] + loss_D['loss_D_fake{}'.format(i)])
                    loss_D['loss_D_sum{}'.format(i)].backward()
                    optim["discriminator"][i].step()

                #backward netG
                set_requires_grad(net["discriminator"], False)
                optim["generator"][0].zero_grad()

                pred_fake = net["discriminator"][0](output)
                #loss
                for i in range(len(net["generator"])):
                    loss_G['loss_G{}'.format(i)] = fn_loss(pred_fake, torch.ones_like(pred_fake))
                    loss_G['loss_G{}'.format(i)].backward()
                    optim["generator"][i].step()


                #손실함수 계산
                loss_G_train += [loss_G['loss_G0'].item()]
                loss_D_real_train += [loss_D['loss_D_real0'].item()]
                loss_D_fake_train += [loss_D['loss_D_fake0'].item()]


                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                        "GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                        (epoch, epochs, batch, num_batch_train,
                        np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))



                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_train * (epoch - 1) + batch

                if epoch % SAVE_NUM == 0 and batch == num_batch_train:
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

                writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            
            
            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            if epoch % SAVE_NUM == 0:
                save(ckpt_dir, net, optim, epoch, isgan=isgan)

        writer_train.close()




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

