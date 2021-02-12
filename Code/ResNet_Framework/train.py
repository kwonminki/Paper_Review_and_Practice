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

#PATH
HOME_PATH = Path("/home/mingi/mingi")
DATA_PATH = HOME_PATH/"data"
PATH = HOME_PATH/"ResNet_Framework"

#Parser
parser = argparse.ArgumentParser(description="Train UNet",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--epochs", default=100, type=int, dest="epochs")

parser.add_argument("--data_dir", default=DATA_PATH/"BSD500/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default=PATH/"checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default=PATH/"log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default=PATH/"result", type=str, dest="result_dir")

parser.add_argument("--device", default='cuda', type=str, dest="device")

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--task", default="super_resolution", choices=["inpainting", "denoising", "super_resolution"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4], dest='opts') #입력으로 리스트 받기

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--norm", default="bnorm", choices=["bnorm", "inorm"], type=str, dest="norm")
#1
parser.add_argument("--network", default="unet", choices=["resnet", "unet", "srresnet"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

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

#모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 재설정.
ckpt_dir = ckpt_dir/task/network/learning_type
log_dir = log_dir/task/network/learning_type
result_dir = result_dir/task/network/learning_type

result_dir_train = result_dir/"train"
result_dir_val = result_dir/"val"
result_dir_test = result_dir/"test"

#결과 저장용 디렉토리 생성
mkdirlist([data_dir,ckpt_dir,log_dir,result_dir, result_dir_test, result_dir_train, result_dir_val])
if not os.path.exists(result_dir_test/'png'):
    os.makedirs(result_dir_test/'png')
    os.makedirs(result_dir_test/'numpy')
    os.makedirs(result_dir_train/'png')
    os.makedirs(result_dir_val/'png')


if mode == 'train':
    #학습데이터
    transform_train = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), RandomFlip()]) #Randomcrop으로 이미지 사이즈 맞춰줌
    transform_val = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5)])

    dataset_train = Dataset(data_dir/'train', transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir/'val', transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    #필요해서 만듬
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    #batch 사이즈 
    num_batch_train = np.ceil(num_data_train/batch_size)
    num_batch_val = np.ceil(num_data_val/batch_size)

elif mode == 'test':
    #테스트 데이터
    transform_test = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5)])

    dataset_test = Dataset(data_dir/'test', transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
    
    #필요해서 만듬
    num_data_test = len(dataset_test)

    #batch 사이즈
    num_batch_test = np.ceil(num_data_test/batch_size)

##
#2
##네트워크 생성
if network == "unet":
    net = UNet(in_channels=nch, out_channels=1, norm=norm, learning_type=learning_type).to(device)
elif network == "srresnet":
    net = SRResNet(in_channels=nch, out_channels=nch , nker=nker, norm=norm, learning_type=learning_type).to(device)
elif network == "resnet":
    net = ResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type, norm=norm, n_blocks=16).to(device)

#loss는 MSE로
fn_loss = nn.MSELoss().to(device)
#fn_loss = nn.BCEWithLogitsLoss().to(device)

#Adam사용
optim = torch.optim.Adam(net.parameters(), lr=lr)

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
        net, optim, st_epoch = load(ckpt_dir, net, optim)

    for epoch in range(st_epoch+1, epochs+1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            #backward
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            loss_arr += [loss.item()]
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
            (epoch, epochs, batch, num_batch_train, np.mean(loss_arr)))

            # 저장 라벨이 클래스가 아니라 이미지이다.
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            #png파일로 저장
            #저장할 이미지를 0~1로
            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train * (epoch - 1) + batch

            if epoch % 2 == 0 and batch == num_batch_train:
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

            

            # writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 5 == 0:
            save(ckpt_dir, net, optim, epoch)


        with torch.no_grad():
            net.eval()
            loss_mse = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_mse += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                        (epoch, epochs, batch, num_batch_val, np.mean(loss_arr)))

                # 저장하기
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))


                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_train * (epoch - 1) + batch

                if epoch % 2 == 0 and batch == num_batch_val:
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)
                # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_mse), epoch)


    writer_train.close()
    writer_val.close()

elif mode == "test":
    ## 네트워크 학습시키기
    st_epoch = 0
    net, optim, st_epoch = load(ckpt_dir, net, optim)

    with torch.no_grad():
        net.eval()
        loss_mse = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass

            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_mse += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                    (batch, num_batch_test, np.mean(loss_mse)))

            # Tensorboard 저장하기
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch -1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_.squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_.squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_.squeeze(), cmap=cmap)

            print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_mse)))

