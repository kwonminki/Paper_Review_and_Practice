#%%
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from pathlib import Path

from model import UNet
from dataset import * #모두다
from util import * #save, load
from mkdir import mkdirlist

#PATH
HOME_PATH = Path("/home/mingi/mingi")
DATA_PATH = HOME_PATH/"data"
PATH = HOME_PATH/"UNet_Module"

#Parser
parser = argparse.ArgumentParser(description="Train UNet",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--epochs", default=100, type=int, dest="epochs")

parser.add_argument("--data_dir", default=DATA_PATH/"ISBI_EM_stacks", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default=PATH/"checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default=PATH/"log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default=PATH/"result", type=str, dest="result_dir")

parser.add_argument("--device", default='cuda', type=str, dest="device")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

#하이퍼파라미터
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir
train_continue = args.train_continue

mkdirlist([data_dir,ckpt_dir,log_dir,result_dir])
if not os.path.exists(result_dir/'png'):
    os.makedirs(result_dir/'png')
    os.makedirs(result_dir/'numpy')

device = args.device
mode = args.mode


if mode == 'train':
    #학습데이터
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir/'train', transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir/'val', transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    #필요해서 만듬
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    #batch 사이즈 
    num_batch_train = np.ceil(num_data_train/batch_size)
    num_batch_val = np.ceil(num_data_val/batch_size)

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

elif mode == 'eval':
    #테스트 데이터
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir/'test', transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    
    #필요해서 만듬
    num_data_test = len(dataset_test)

    #batch 사이즈 
    num_batch_test = np.ceil(num_data_test/batch_size)

##네트워크 생성
net = UNet().to(device)
#loss는 BCE로
fn_loss = nn.BCEWithLogitsLoss().to(device)
#Adam사용
optim = torch.optim.Adam(net.parameters(), lr=lr)

#다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_binaryclass = lambda x: 1.0*(x>0.5)


if mode == "train":
    ## 네트워크 학습시키기
    st_epoch = 0
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

            # 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_binaryclass(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 5 == 0:
            save(ckpt_dir, net, optim, epoch)


        with torch.no_grad():
                net.eval()
                loss_arr = []

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)

                    output = net(input)

                    # 손실함수 계산하기
                    loss = fn_loss(output, label)

                    loss_arr += [loss.item()]

                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                            (epoch, epochs, batch, num_batch_val, np.mean(loss_arr)))

                    # Tensorboard 저장하기
                    label = fn_tonumpy(label)
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output = fn_tonumpy(fn_binaryclass(output))

                    writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

                writer_val.add_scalar('loss', np.mean(loss_arr), epoch)


    writer_train.close()
    writer_val.close()

elif mode == "eval":
    ## 네트워크 학습시키기
    st_epoch = 0
    net, optim, st_epoch = load(ckpt_dir, net, optim)

    with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_test, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                        (batch, num_batch_test, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_binaryclass(output))

                for j in range(label.shape[0]):
                    id = num_batch_test * (batch -1) + j

                    plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                    np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                    np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                    np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
            (batch, num_batch_test, np.mean(loss_arr)))

