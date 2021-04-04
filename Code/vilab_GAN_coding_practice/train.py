#%%
import argparse
import wandb

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

from tqdm import tqdm

def train(args):
    #하이퍼파라미터
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    norm = args.norm

    #디렉토리
    data_dir = args.data_dir
    project_dir = args.project_dir

    project_dir = project_dir/args.network/args.advloss/args.foldername
    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 설정.
    ckpt_dir = project_dir/"ckpts"
    log_dir = project_dir/"logs"
    result_dir = project_dir/"fake_images/train"

    #결과 저장용 디렉토리 생성
    mkdirlist([ckpt_dir,log_dir,result_dir])

    #args
    task = args.task
    opts = args.opts

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    train_continue = args.train_continue
    network = args.network

    device = args.device

    PRINT_LOSS_NUM = args.print_every # 한 epoch에 print 몇번 할지
    SAVE_SAMPLE_EVERY = args.sample_every # epoch 몇번마다 image save 할지 sample_every
    SAVE_MODEL_EVERY = args.chpt_every # epoch 몇번마다 model save 할지 chpt_every
    SAVE_IMAGE_NUM = args.save_image_num #image 몇개씩 저장할지


    #학습데이터
    transform_train = transforms.Compose([transforms.Resize((ny, nx)), transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    dataset_train = Dataset(data_dir, transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8) #num_workers=8

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

        net = {"generator" : netG, "discriminator" : netD}


    #loss는 MSE로
    fn_loss = nn.BCELoss().to(device)
    #fn_loss = nn.BCEWithLogitsLoss().to(device)

    #Adam사용
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999)) #논문에서 변경함.
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optim = {"generator" : optimG, "discriminator" : optimD}

    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean
    #fn_binaryclass = lambda x: 1.0*(x>0.5)

    wandb.init(project='DCGAN_{}'.format(args.foldername), config=args)

    
    ## 네트워크 학습시키기
    st_epoch = 0
    #저장해둔 네트워크 불러오기.
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir, net, optim)

    for epoch in range(st_epoch+1, epochs+1):
        
        loss_D = {}
        loss_G = {}
        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        net["generator"].train()
        loss_G.update({'loss_G' : []})
        
        net["discriminator"].train()
        loss_D.update({'loss_D_real' : []})
        loss_D.update({'loss_D_fake' : []})

        print("Train EPOCH : {} / {}".format(epoch, epochs))  
        for batch, data in enumerate(tqdm(loader_train), 1):
            label = data['label'].to(device)

            input = torch.randn(label.shape[0], 100, 1, 1).to(device=device, dtype=torch.float)
            output = net["generator"](input)

            #backward netD
            set_requires_grad(net["discriminator"], True)
            optim["discriminator"].zero_grad()

            pred_real = net["discriminator"](label)
            pred_fake = net["discriminator"](output.detach())  #연결 끊음. - detach

            #loss..
            loss_D['loss_D_real'] = fn_loss(pred_real, torch.ones_like(pred_real))
            loss_D['loss_D_fake'] = fn_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D['loss_D_sum'] = 0.5 * (loss_D['loss_D_real'] + loss_D['loss_D_fake'])
            loss_D['loss_D_sum'].backward()
            optim["discriminator"].step()

            #backward netG
            set_requires_grad(net["discriminator"], False)
            optim["generator"].zero_grad()

            pred_fake = net["discriminator"](output)
            #loss

            loss_G['loss_G'] = fn_loss(pred_fake, torch.ones_like(pred_fake))
            loss_G['loss_G'].backward()
            optim["generator"].step()


            #손실함수 계산
            loss_G_train += [loss_G['loss_G'].item()]
            loss_D_real_train += [loss_D['loss_D_real'].item()]
            loss_D_fake_train += [loss_D['loss_D_fake'].item()]

            #화면에 출력 PRINT_LOSS_NUM에 맞게 띄엄띄엄 출력.
            if batch%(num_batch_train//PRINT_LOSS_NUM) == 0:
                print("-------- | GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                (np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))
            
            wandb.log({"GEN_loss" : loss_G['loss_G'], "DISC_real_loss" : loss_D['loss_D_real'], "DISC_fake_loss" : loss_D['loss_D_fake'], "DISC_loss" : loss_D['loss_D_sum']})

            #이미지 저장. SAVE_SAMPLE_EVERY에 맞게 띄엄띄엄 저장.
            if epoch % SAVE_SAMPLE_EVERY == 0:
                if batch%(num_batch_train//SAVE_IMAGE_NUM) == 0:
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
                    output = np.clip(output, a_min=0, a_max=1)
                    id = num_batch_train * (epoch - 1) + batch
                    i = int(batch//(num_batch_train//SAVE_IMAGE_NUM))
                    plt.imsave(os.path.join(result_dir, '{}_{}_fakeimage.png'.format(id, i)), output[i].squeeze())

        print("TRAIN: EPOCH %04d / %04d | "
        "GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
        (epoch, epochs, np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))


        if epoch % SAVE_MODEL_EVERY == 0:
            save(ckpt_dir, net, optim, epoch)


    wandb.finish()

