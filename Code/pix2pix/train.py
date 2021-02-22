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

    wgt = args.wgt

    cmap = None #"gray" 하면 흑백으로 저장됨
    isgan = "yes" # GAN인지
    SAVE_NUM = 50 # epoch 몇번마다 save 할지

    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 재설정.
    ckpt_dir = ckpt_dir/task/network/learning_type
    log_dir = log_dir/task/network/learning_type
    result_dir = result_dir/task/network/learning_type
    result_dir_val = result_dir/"val"
    result_dir_train = result_dir/"train"

    #결과 저장용 디렉토리 생성
    mkdirlist([data_dir,ckpt_dir,log_dir,result_dir, result_dir_train, result_dir_val])
    if not os.path.exists(result_dir_train/'png'):
        os.makedirs(result_dir_train/'png')
        os.makedirs(result_dir_val/'png')



    if mode == 'train':
        #학습데이터 #Resize로 286x286로 만들고 Randomcrop으로 이미지 사이즈 맞춰줌(Jitter)
        transform_train = transforms.Compose([Resize(shape=(286,286, nch)), RandomCrop((nx,ny)), Normalization(mean=0.5, std=0.5)]) 
        transform_val = transforms.Compose([Resize(shape=(286,286, nch)), RandomCrop((nx,ny)), Normalization(mean=0.5, std=0.5)])
        print(opts)
        dataset_train = Dataset(data_dir/"train", transform=transform_train, task=task, opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        #필요해서 만듬
        num_data_train = len(dataset_train)

        #batch 사이즈 
        num_batch_train = np.ceil(num_data_train/batch_size)

        #val데이터
        dataset_val = Dataset(data_dir/"val", transform=transform_val, task=task, opts=opts)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)
        num_data_val = len(dataset_val)
        num_batch_val = np.ceil(num_data_val/batch_size)

    ##
    #2
    ##네트워크 생성 -> GAN 용으로 바꿈. 나중엔 generator, discriminator도 여러개가 될 수 있으므로 그에 맞게 만듬.
    if network == "dcgan":
        netG = DCGAN_Generator(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = DCGAN_Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        net = {"generator" : [netG], "discriminator" : [netD]}
    elif network == "pix2pix":
        netG = Pix2Pix_Generator(in_channels=nch, out_channels=nch, nker=nker).to(device)
        netD = pix2pix_Discriminator(in_channels=2*nch, out_channels=nch, nker=nker).to(device) #input에 Conditional GAN 라벨이 같이 들어간다.

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        net = {"generator" : [netG], "discriminator" : [netD]}

    #loss는 L1 loss 와 GAN loss
    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
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
                for i in range(len(net["generator"])):
                    net["generator"][i].train()
                    loss_G.update({'loss_G_l1_train{}'.format(i) : []})
                    loss_G.update({'loss_G_gan_train{}'.format(i) : []})
                for i in range(len(net["discriminator"])):
                    net["discriminator"][i].train()
                    loss_D.update({'loss_D_real_train{}'.format(i) : []})
                    loss_D.update({'loss_D_fake_train{}'.format(i) : []})
            else:
                loss_arr = []

            for batch, data in enumerate(loader_train, 1):
                label = data['label'].to(device)
                
                input = data['input'].to(device)

                output = net["generator"][0](input)

                #backward netD
                set_requires_grad(net["discriminator"], True)
                optim["discriminator"][0].zero_grad()

                real = torch.cat([input, label], dim=1)
                fake = torch.cat([input, output], dim=1)

                pred_real = net["discriminator"][0](real)
                pred_fake = net["discriminator"][0](fake.detach())  #연결 끊음. - detach

                #loss..
                for i in range(len(net["discriminator"])):
                    loss_D['loss_D_real{}'.format(i)] = fn_gan(pred_real, torch.ones_like(pred_real))
                    loss_D['loss_D_fake{}'.format(i)] = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                    loss_D['loss_D_sum{}'.format(i)] = 0.5 * (loss_D['loss_D_real{}'.format(i)] + loss_D['loss_D_fake{}'.format(i)])
                    loss_D['loss_D_sum{}'.format(i)].backward()
                    optim["discriminator"][i].step()

                #backward netG
                set_requires_grad(net["discriminator"], False)
                optim["generator"][0].zero_grad()

                fake = torch.cat([input, output], dim=1)
                pred_fake = net["discriminator"][0](fake)
                #loss
                for i in range(len(net["generator"])):
                    loss_G['loss_G_gan{}'.format(i)] = fn_gan(pred_fake, torch.ones_like(pred_fake))
                    loss_G['loss_G_l1{}'.format(i)] = fn_l1(output, label) #output과 label의 거리.
                    loss_G['loss_G_combination{}'.format(i)] = loss_G['loss_G_gan{}'.format(i)] + (wgt * loss_G['loss_G_l1{}'.format(i)])
                    loss_G['loss_G_combination{}'.format(i)].backward()
                    optim["generator"][i].step()


                #손실함수 계산
                for i in range(len(net["generator"])):
                    loss_G['loss_G_gan_train{}'.format(i)] += [loss_G['loss_G_gan{}'.format(i)].item()]
                    loss_G['loss_G_l1_train{}'.format(i)] += [loss_G['loss_G_l1{}'.format(i)].item()]
                for i in range(len(net["discriminator"])):
                    loss_D['loss_D_real_train{}'.format(i)] += [loss_D['loss_D_real{}'.format(i)].item()]
                    loss_D['loss_D_fake_train{}'.format(i)] += [loss_D['loss_D_fake{}'.format(i)].item()]


                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                        "GEN %.4f | GEN GAN %.4f | "
                        "DISC REAL: %.4f | DISC FAKE: %.4f" %
                        (epoch, epochs, batch, num_batch_train,
                        np.mean(loss_G["loss_G_l1_train0"]), np.mean(loss_G["loss_G_gan_train0"]), np.mean(loss_D["loss_D_real_train0"]), np.mean(loss_D["loss_D_fake_train0"])))

                #tensorboard
                if batch % 20 == 0:
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                    label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                    input = np.clip(input, a_min=0, a_max=1)
                    label = np.clip(label, a_min=0, a_max=1)
                    output = np.clip(output, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                    writer_train.add_image('input', input, id, dataformats='NHWC')
                    writer_train.add_image('label', label, id, dataformats='NHWC')
                    writer_train.add_image('output', output, id, dataformats='NHWC')
            
            
            writer_train.add_scalar('loss_G', np.mean(loss_G["loss_G_l1_train0"]), epoch)
            writer_train.add_scalar('loss_G', np.mean(loss_G["loss_G_gan_train0"]), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D["loss_D_real_train0"]), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D["loss_D_fake_train0"]), epoch)

            #val
            with torch.no_grad():

                if isgan is not None:
                    loss_D = {}
                    loss_G = {}
                    for i in range(len(net["generator"])):
                        net["generator"][i].eval()
                        loss_G.update({'loss_G_l1_val{}'.format(i) : []})
                        loss_G.update({'loss_G_gan_val{}'.format(i) : []})
                    for i in range(len(net["discriminator"])):
                        net["discriminator"][i].eval()
                        loss_D.update({'loss_D_real_val{}'.format(i) : []})
                        loss_D.update({'loss_D_fake_val{}'.format(i) : []})
                else:
                    loss_arr = []

                for batch, data in enumerate(loader_val, 1):
                    label = data['label'].to(device)
                    
                    input = data['input'].to(device)

                    output = net["generator"][0](input)

                    #backward netD
                    # set_requires_grad(net["discriminator"], True)
                    # optim["discriminator"][0].zero_grad()

                    real = torch.cat([input, label], dim=1)
                    fake = torch.cat([input, output], dim=1)

                    pred_real = net["discriminator"][0](real)
                    pred_fake = net["discriminator"][0](fake.detach())  #연결 끊음. - detach

                    #loss..
                    for i in range(len(net["discriminator"])):
                        loss_D['loss_D_real{}'.format(i)] = fn_gan(pred_real, torch.ones_like(pred_real))
                        loss_D['loss_D_fake{}'.format(i)] = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                        loss_D['loss_D_sum{}'.format(i)] = 0.5 * (loss_D['loss_D_real{}'.format(i)] + loss_D['loss_D_fake{}'.format(i)])
                        # loss_D['loss_D_sum{}'.format(i)].backward()
                        # optim["discriminator"][i].step()

                    #backward netG
                    # set_requires_grad(net["discriminator"], False)
                    # optim["generator"][0].zero_grad()

                    fake = torch.cat([input, output], dim=1)
                    pred_fake = net["discriminator"][0](fake)
                    #loss
                    for i in range(len(net["generator"])):
                        loss_G['loss_G_gan{}'.format(i)] = fn_gan(pred_fake, torch.ones_like(pred_fake))
                        loss_G['loss_G_l1{}'.format(i)] = fn_l1(output, label) #output과 label의 거리.
                        loss_G['loss_G_combination{}'.format(i)] = loss_G['loss_G_gan{}'.format(i)] + (wgt * loss_G['loss_G_l1{}'.format(i)])
                        # loss_G['loss_G_combination{}'.format(i)].backward()
                        # optim["generator"][i].step()


                    #손실함수 계산
                    for i in range(len(net["generator"])):
                        loss_G['loss_G_gan_val{}'.format(i)] += [loss_G['loss_G_gan{}'.format(i)].item()]
                        loss_G['loss_G_l1_val{}'.format(i)] += [loss_G['loss_G_l1{}'.format(i)].item()]
                    for i in range(len(net["discriminator"])):
                        loss_D['loss_D_real_val{}'.format(i)] += [loss_D['loss_D_real{}'.format(i)].item()]
                        loss_D['loss_D_fake_val{}'.format(i)] += [loss_D['loss_D_fake{}'.format(i)].item()]


                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | "
                            "GEN %.4f | GEN GAN %.4f | "
                            "DISC REAL: %.4f | DISC FAKE: %.4f" %
                            (epoch, epochs, batch, num_batch_train,
                            np.mean(loss_G["loss_G_l1_val0"]), np.mean(loss_G["loss_G_gan_val0"]), np.mean(loss_D["loss_D_real_val0"]), np.mean(loss_D["loss_D_fake_val0"])))

                    #tensorboard
                    if batch % 5 == 0:
                        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                        label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                        output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                        input = np.clip(input, a_min=0, a_max=1)
                        label = np.clip(label, a_min=0, a_max=1)
                        output = np.clip(output, a_min=0, a_max=1)

                        id = num_batch_train * (epoch - 1) + batch

                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                        writer_val.add_image('input', input, id, dataformats='NHWC')
                        writer_val.add_image('label', label, id, dataformats='NHWC')
                        writer_val.add_image('output', output, id, dataformats='NHWC')
                
                
                writer_val.add_scalar('loss_G', np.mean(loss_G["loss_G_l1_val0"]), epoch)
                writer_val.add_scalar('loss_G', np.mean(loss_G["loss_G_gan_val0"]), epoch)
                writer_val.add_scalar('loss_D_real', np.mean(loss_D["loss_D_real_val0"]), epoch)
                writer_val.add_scalar('loss_D_fake', np.mean(loss_D["loss_D_fake_val0"]), epoch)



            if epoch % SAVE_NUM == 0:
                save(ckpt_dir, net, optim, epoch, isgan=isgan)

        writer_val.close()
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

    wgt = args.wgt

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

        dataset_test = Dataset(data_dir/"test", transform=transform_test, task=task, opts=opts)
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
    elif network == "pix2pix":
        netG = Pix2Pix_Generator(in_channels=nch, out_channels=nch, nker=nker).to(device)
        netD = pix2pix_Discriminator(in_channels=2*nch, out_channels=nch, nker=nker).to(device) #input에 Conditional GAN 라벨이 같이 들어간다.

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        net = {"generator" : [netG], "discriminator" : [netD]}


    #loss는 L1 loss 와 GAN loss
    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)

    #Adam사용
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999)) #논문에서 변경함.
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optim = {"generator" : [optimG], "discriminator" : [optimD]}

    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean
    fn_binaryclass = lambda x: 1.0*(x>0.5)


    if mode == "test":
        ## 네트워크 학습시키기
        st_epoch = 0
        net, optim, st_epoch = load(ckpt_dir, net, optim, isgan=isgan)

        with torch.no_grad():

            for epoch in range(st_epoch+1, epochs+1):
                if isgan is not None:
                    loss_G = {}
                    for i in range(len(net["generator"])):
                        net["generator"][i].eval()
                        loss_G.update({'loss_G_l1_val{}'.format(i) : []})
                        loss_G.update({'loss_G_gan_val{}'.format(i) : []})
                else:
                    loss_arr = []

                for batch, data in enumerate(loader_test, 1):
                    label = data['label'].to(device)
                    
                    input = data['input'].to(device)

                    output = net["generator"][0](input)

                    #loss
                    for i in range(len(net["generator"])):
                        loss_G['loss_G_l1{}'.format(i)] = fn_l1(output, label) #output과 label의 거리.


                    #손실함수 계산
                    for i in range(len(net["generator"])):
                        loss_G['loss_G_l1_val{}'.format(i)] += [loss_G['loss_G_l1{}'.format(i)].item()]


                    print("TEST: BATCH %04d / %04d | GEN %.4f" %
                            (batch, num_batch_test, np.mean(loss_G["loss_G_l1_val0"])))

                    # Tensorboard 저장하기
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                    label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                    for j in range(label.shape[0]):

                        id = batch_size * (batch - 1) + j

                        input_ = input[j]
                        label_ = label[j]
                        output_ = output[j]

                        np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                        np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                        np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                        input_ = np.clip(input_, a_min=0, a_max=1)
                        label_ = np.clip(label_, a_min=0, a_max=1)
                        output_ = np.clip(output_, a_min=0, a_max=1)

                        plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                        plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                        plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

                print('AVERAGE TEST: GEN L1 %.4f' % np.mean(loss_G_l1_test))