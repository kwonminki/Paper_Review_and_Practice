#%%
import argparse

import os
from pathlib import Path
import numpy as np

import itertools

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
    #opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]
    opts = args.opts

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    train_continue = args.train_continue
    network = args.network
    learning_type = args.learning_type

    device = args.device
    mode = args.mode

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident

    cmap = None #"gray" 하면 흑백으로 저장됨
    isgan = "yes" # GAN인지
    SAVE_NUM = 2 # epoch 몇번마다 save 할지

    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 재설정.
    ckpt_dir = ckpt_dir/task/network/learning_type/norm
    log_dir = log_dir/task/network/learning_type/norm
    result_dir = result_dir/task/network/learning_type/norm
    result_dir_train = result_dir/"train"

    #결과 저장용 디렉토리 생성
    mkdirlist([data_dir,ckpt_dir,log_dir,result_dir, result_dir_train])
    if not os.path.exists(result_dir_train/'png'):
        os.makedirs(result_dir_train/'png')



    if mode == 'train':
        #학습데이터 #Resize로 286x286로 만들고 Randomcrop으로 이미지 사이즈 맞춰줌(Jitter)
        transform_train = transforms.Compose([Resize(shape=(286,286, nch)), RandomCrop((nx,ny)), Normalization(mean=0.5, std=0.5)])

        dataset_train = Dataset(data_dir/"train", transform=transform_train, data_type='both')
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
        netD = DCGAN_Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        net = {"generator" : [netG], "discriminator" : [netD]}
    elif network == "pix2pix":
        netG = Pix2Pix_Generator(in_channels=nch, out_channels=nch, nker=nker).to(device)
        netD = pix2pix_Discriminator(in_channels=2*nch, out_channels=nch, nker=nker).to(device) #input에 Conditional GAN 라벨이 같이 들어간다.

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        net = {"generator" : [netG], "discriminator" : [netD]}
    elif network == "cyclegan":
        netG_X2Y = CycleGAN_G(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netG_Y2X = CycleGAN_G(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)

        init_weights(netG_X2Y, init_type='normal', init_gain=0.02)
        init_weights(netG_Y2X, init_type='normal', init_gain=0.02)

        netD_X = CycleGAN_D(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)
        netD_Y = CycleGAN_D(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netD_X, init_type='normal', init_gain=0.02)
        init_weights(netD_Y, init_type='normal', init_gain=0.02)

        net = {"generator":[netG_X2Y, netG_Y2X], "discriminator":[netD_X, netD_Y]}



    #loss는 L1 loss 와 GAN loss
    fn_cycle = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    fn_ident = nn.L1Loss().to(device)

    #Adam사용
    optimG = torch.optim.Adam(itertools.chain(netG_X2Y.parameters(), netG_Y2X.parameters()), lr=lr, betas=(0.5, 0.999)) #논문에서 변경함.
    optimD = torch.optim.Adam(itertools.chain(netD_X.parameters(), netD_Y.parameters()), lr=lr, betas=(0.5, 0.999))
    optim = {"generator" : optimG, "discriminator" : optimD}

    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean
    #fn_binaryclass = lambda x: 1.0*(x>0.5)

    # writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))


    if mode == "train":
        ## 네트워크 학습시키기
        st_epoch = 0
        #저장해둔 네트워크 불러오기.
        if train_continue == 'on':
            net, optim, st_epoch = load(ckpt_dir, net, optim, isgan=isgan)

        for epoch in range(st_epoch+1, epochs+1):
            
            for i in range(len(net["generator"])):
                net["generator"][i].train()
            for i in range(len(net["discriminator"])):
                net["discriminator"][i].train()
            
            loss_G_X2Y_train = []
            loss_G_Y2X_train = []

            loss_D_X_train = []
            loss_D_Y_train = []
            
            loss_cycle_X_train = []
            loss_cycle_Y_train = []

            loss_ident_X_train = []
            loss_ident_Y_train = []


            for batch, data in enumerate(loader_train, 1):
       
                input_X = data['data_a'].to(device)
                input_Y = data['data_b'].to(device)

                #X도메인에서 Y도메인으로 trans하는 forward
                output_Y = net["generator"][0](input_X) #X2Y
                cycle_result_X = net["generator"][1](output_Y)

                output_X = net["generator"][1](input_Y) #Y2X
                cycle_result_Y = net["generator"][0](output_X)

                #backward netD
                set_requires_grad(net["discriminator"], True)
                optim["discriminator"].zero_grad()

                #backward netD_X
                pred_real_X = net["discriminator"][0](input_X)
                pred_fake_X = net["discriminator"][0](output_X.detach())  #연결 끊음. - detach

                #loss..
                loss_D_X_real = fn_gan(pred_real_X, torch.ones_like(pred_real_X))
                loss_D_X_fake = fn_gan(pred_real_X, torch.zeros_like(pred_real_X))
                loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5

                #backward netD_Y
                pred_real_Y = net["discriminator"][1](input_Y)
                pred_fake_Y = net["discriminator"][1](output_Y.detach())

                #loss
                loss_D_Y_real = fn_gan(pred_real_Y, torch.ones_like(pred_real_Y))
                loss_D_Y_fake = fn_gan(pred_real_Y, torch.zeros_like(pred_real_Y))
                loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5

                loss_D = loss_D_X + loss_D_Y

                loss_D.backward()
                optim["discriminator"].step()
                #End backward netD

                #backward netG Generator 해야하니까 이제 discriminator requires false로 바꿔줘야함.
                set_requires_grad(net["discriminator"], False)
                optim["generator"].zero_grad()
                
                #이번에는 detach 를 안해준다.
                pred_fake_X = net["discriminator"][0](output_X) # netD_X
                pred_fake_Y = net["discriminator"][1](output_Y) # netD_Y

                loss_G_X2Y = fn_gan(pred_fake_X, torch.ones_like(pred_fake_X)) #Generator가 속여야 하므로 ones
                loss_G_Y2X = fn_gan(pred_fake_Y, torch.ones_like(pred_fake_Y))

                #loss cycle
                loss_cycle_X = fn_cycle(cycle_result_X, input_X)
                loss_cycle_Y = fn_cycle(cycle_result_Y, input_Y)
                
                #loss ident -> Photo generation from paintings (painting->photo)에만 적용.
                identX = net["generator"][1](input_X) #Y2X 에다가 X를 넣었을때는 그냥 X가 나와야 함.
                identY = net["generator"][0](input_Y) #vise varsa
                                
                loss_ident_X = fn_ident(identX, input_X)
                loss_ident_Y = fn_ident(identY, input_Y)

                loss_G = (loss_G_X2Y + loss_G_Y2X) + wgt_cycle*(loss_cycle_X + loss_cycle_Y) + wgt_ident*(loss_ident_X + loss_ident_Y)
                
                loss_G.backward()
                optim["generator"].step()


                #손실함수 계산
                loss_D_X_train += [loss_D_X.item()]
                loss_D_Y_train += [loss_D_Y.item()]

                loss_G_X2Y_train += [loss_G_X2Y.item()]
                loss_G_Y2X_train += [loss_G_Y2X.item()]

                loss_cycle_X_train += [loss_cycle_X.item()]
                loss_cycle_Y_train += [loss_cycle_Y.item()]

                loss_ident_X_train += [loss_ident_X.item()]
                loss_ident_Y_train += [loss_ident_Y.item()]


                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                        "GEN : X2Y %.4f , Y2X %.4f | "
                        "Cycle : X %.4f , Y %.4f | "
                        "Ident : X %.4f , Y %.4f | "
                        "DISC : X %.4f , Y %.4f" %
                        (epoch, epochs, batch, num_batch_train,
                        np.mean(loss_G_X2Y_train), np.mean(loss_G_Y2X_train),
                        np.mean(loss_cycle_X_train), np.mean(loss_cycle_Y_train),
                        np.mean(loss_ident_X_train), np.mean(loss_ident_Y_train),
                        np.mean(loss_D_X_train), np.mean(loss_D_Y_train)                        
                        ))

                #tensorboard
                if batch % 100 == 0 and epoch % SAVE_NUM == 0:
                    input_X = fn_tonumpy(fn_denorm(input_X, mean=0.5, std=0.5)).squeeze()
                    input_Y = fn_tonumpy(fn_denorm(input_Y, mean=0.5, std=0.5)).squeeze()
                    output_X = fn_tonumpy(fn_denorm(output_X, mean=0.5, std=0.5)).squeeze()
                    output_Y = fn_tonumpy(fn_denorm(output_Y, mean=0.5, std=0.5)).squeeze()

                    input_X = np.clip(input_X, a_min=0, a_max=1)
                    input_Y = np.clip(input_Y, a_min=0, a_max=1)
                    output_X = np.clip(output_X, a_min=0, a_max=1)
                    output_Y = np.clip(output_Y, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_X.png' % id), input_X[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_Y.png' % id), input_Y[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_X.png' % id), output_X[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_Y.png' % id), output_Y[0], cmap=cmap)


                    # writer_train.add_image('input', input, id, dataformats='NHWC')
                    # writer_train.add_image('label', label, id, dataformats='NHWC')
                    # writer_train.add_image('output', output, id, dataformats='NHWC')
            
            
            # writer_train.add_scalar('loss_G', np.mean(loss_G["loss_G_l1_train0"]), epoch)
            # writer_train.add_scalar('loss_G', np.mean(loss_G["loss_G_gan_train0"]), epoch)
            # writer_train.add_scalar('loss_D_real', np.mean(loss_D["loss_D_real_train0"]), epoch)
            # writer_train.add_scalar('loss_D_fake', np.mean(loss_D["loss_D_fake_train0"]), epoch)

            if epoch % SAVE_NUM == 0:
                save(ckpt_dir, net, optim, epoch, isgan=isgan)

        # writer_train.close()



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

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident

    train_continue = args.train_continue
    network = args.network
    learning_type = args.learning_type

    device = args.device
    mode = args.mode

    cmap = None #"gray" 하면 흑백으로 저장됨
    isgan = "yes" # GAN인지

    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 재설정.
    ckpt_dir = ckpt_dir/task/network/learning_type
    log_dir = log_dir/task/network/learning_type
    result_dir = result_dir/task/network/learning_type

    result_dir_test = result_dir/"test"

    #결과 저장용 디렉토리 생성
    mkdirlist([data_dir,ckpt_dir,log_dir,result_dir, result_dir_test])
    if not os.path.exists(result_dir_test/'png'):
        os.makedirs(result_dir_test/'png')




    if mode == 'test':
        #테스트 데이터
        transform_test = transforms.Compose([Resize(shape=(ny,nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_test_X = Dataset(data_dir/"testA", transform=transform_test, data_type='a')
        dataset_test_Y = Dataset(data_dir/"testB", transform=transform_test, data_type='b')
        loader_test_X = DataLoader(dataset_test_X, batch_size=batch_size, shuffle=False, num_workers=8)
        loader_test_Y = DataLoader(dataset_test_Y, batch_size=batch_size, shuffle=False, num_workers=8)
        
        #필요해서 만듬
        num_data_test_X = len(dataset_test_X)
        num_data_test_Y = len(dataset_test_Y)

        #batch 사이즈
        num_batch_test_X = np.ceil(num_data_test_X/batch_size)
        num_batch_test_Y = np.ceil(num_data_test_Y/batch_size)


    ##
    #2
    ##네트워크 생성 -> GAN 용으로 바꿈. 나중엔 generator, discriminator도 여러개가 될 수 있으므로 그에 맞게 만듬.
    if network == "cyclegan":
        netG_X2Y = CycleGAN_G(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netG_Y2X = CycleGAN_G(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)

        init_weights(netG_X2Y, init_type='normal', init_gain=0.02)
        init_weights(netG_Y2X, init_type='normal', init_gain=0.02)

        netD_X = CycleGAN_D(in_channels=nch, out_channels=1, nker=nker, norm=norm)
        netD_Y = CycleGAN_D(in_channels=nch, out_channels=1, nker=nker, norm=norm)

        init_weights(netD_X, init_type='normal', init_gain=0.02)
        init_weights(netD_Y, init_type='normal', init_gain=0.02)

        net = {"generator":[netG_X2Y, netG_Y2X], "discriminator":[netD_X, netD_Y]}

    #loss는 L1 loss 와 GAN loss
    fn_cycle = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    fn_ident = nn.L1Loss().to(device)

    #Adam사용
    optimG = torch.optim.Adam(itertools.chain(netG_X2Y.parameters, netG_Y2X.parameters), lr=lr, betas=(0.5, 0.999)) #논문에서 변경함.
    optimD = torch.optim.Adam(itertools.chain(netD_X.parameters, netD_Y.parameters), lr=lr, betas=(0.5, 0.999))
    optim = {"generator" : optimG, "discriminator" : optimD}

    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean
    fn_binaryclass = lambda x: 1.0*(x>0.5)


    if mode == "test":
        ## 네트워크 학습시키기
        st_epoch = 0
        net, optim, st_epoch = load(ckpt_dir, net, optim, isgan=isgan)

        with torch.no_grad():
            if isgan is not None:
                for i in range(len(net["generator"])):
                    net["generator"][i].eval()
                
            for batch, data in enumerate(loader_test_X, 1):
                
                input_X = data['data_a'].to(device)                   
                output_Y = net["generator"][0](input_X)
                
                input_X = fn_tonumpy(fn_denorm(input_X, mean=0.5, std=0.5)).squeeze()
                output_Y = fn_tonumpy(fn_denorm(output_Y, mean=0.5, std=0.5)).squeeze()

                # Tensorboard 저장하기
                # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                for j in range(input_X.shape[0]):

                    id = batch_size * (batch - 1) + j

                    input_ = input_X[j]
                    output_ = output_Y[j]

                    input_ = np.clip(input_, a_min=0, a_max=1)
                    output_ = np.clip(output_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input_a.png' % id), input_, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output_b.png' % id), output_, cmap=cmap)


            for batch, data in enumerate(loader_test_Y, 1):
                
                input_Y = data['data_b'].to(device)                   
                output_X = net["generator"][1](input_Y)
                
                input_Y = fn_tonumpy(fn_denorm(input_Y, mean=0.5, std=0.5)).squeeze()
                output_X = fn_tonumpy(fn_denorm(output_X, mean=0.5, std=0.5)).squeeze()

                # Tensorboard 저장하기
                # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                for j in range(input_Y.shape[0]):

                    id = batch_size * (batch - 1) + j

                    input_ = input_Y[j]
                    output_ = output_X[j]

                    input_ = np.clip(input_, a_min=0, a_max=1)
                    output_ = np.clip(output_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input_b.png' % id), input_, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output_a.png' % id), output_, cmap=cmap)


