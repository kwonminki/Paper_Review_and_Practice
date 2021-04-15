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
from ResNet import *
from data_loader import *
from LR_Scheduler import LR_Scheduler

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

    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]

    #학습데이터
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_loader, validation_loader = get_train_valid_loader(data_dir=data_dir,batch_size=batch_size, train_transform=transform, valid_transform=transform)

    # #필요해서 만듬
    # num_data_train = len(dataset_train)

    # #batch 사이즈 
    # num_batch_train = np.ceil(num_data_train/batch_size)

    ##
    #2
    ##네트워크 생성 -> GAN 용으로 바꿈. 나중엔 generator, discriminator도 여러개가 될 수 있으므로 그에 맞게 만듬.
    if network == "resnet18":
        net = resnet18().to(device)

    #loss는 MSE로
    fn_loss = nn.CrossEntropyLoss().to(device)
    #fn_loss = nn.BCEWithLogitsLoss().to(device)

    parameters = [{
        'name': 'base',
        'params': net.parameters(),
        'lr': lr
    }]

    #optimizer
    optim = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = LR_Scheduler(
        optim,
        warmup_epochs = 10, warmup_lr=0, 
        num_epochs=epochs, base_lr=0.03, final_lr=0, 
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )



    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_pred = lambda output: torch.softmax(output, dim=1)
    fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

    #fn_binaryclass = lambda x: 1.0*(x>0.5)

    wandb.init(project='vilab_SSL_coding_practice_{}'.format(args.foldername), config=args, name="{}_{}".format(args.foldername, args.network))

    
    ## 네트워크 학습시키기
    st_epoch = 0
    #저장해둔 네트워크 불러오기.
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir, net, optim)

    for epoch in range(st_epoch+1, epochs+1):
        
        net.train()

        loss_arr = []
        acc_arr = []


        print("EPOCH : {} / {}".format(epoch, epochs))  
        for batch, data in enumerate(tqdm(train_loader), 1):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = net(inputs)
            pred = fn_pred(output)

            optim.zero_grad()

            loss = fn_loss(output, labels)
            acc = fn_acc(pred, labels)

            loss.backward()
            optim.step()
            lr_scheduler.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

        loss_arr_val = []
        acc_arr_val = []

        for batch, data in enumerate(tqdm(validation_loader), 1):
            net.eval()

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = net(inputs)
            pred = fn_pred(output)

            loss_val = fn_loss(output, labels)
            acc_val = fn_acc(pred, labels)

            loss_arr_val += [loss_val.item()]
            acc_arr_val += [acc_val.item()]
        
        wandb.log({"loss_train" : np.mean(loss_arr), "acc_train" : np.mean(acc_arr), "loss_val" : np.mean(loss_arr_val), "acc_val" : np.mean(acc_arr_val), "epoch" : epoch})

        print()
        print('TRAIN: LOSS: %.4f | ACC %.4f' %
                (epoch, epochs, np.mean(loss_arr), np.mean(acc_arr)))
        print('VAL: LOSS: %.4f | ACC %.4f' % (np.mean(loss_arr_val), np.mean(acc_arr_val)))


        if epoch % SAVE_MODEL_EVERY == 0:
            save(ckpt_dir, net, optim, epoch)


    wandb.finish()

