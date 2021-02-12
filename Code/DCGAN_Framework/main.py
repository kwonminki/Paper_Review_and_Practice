import argparse

import os
from pathlib import Path

from train import *

#PATH
HOME_PATH = Path("/home/mingi/mingi")
DATA_PATH = HOME_PATH/"data"
PATH = HOME_PATH/"DCGAN_Framework"

#Parser
parser = argparse.ArgumentParser(description="Train UNet",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--epochs", default=10, type=int, dest="epochs")

parser.add_argument("--data_dir", default=DATA_PATH/"CELEBA/img_align_celeba", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default=PATH/"checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default=PATH/"log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default=PATH/"result", type=str, dest="result_dir")

parser.add_argument("--device", default='cuda', type=str, dest="device")

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--task", default="dcgan", choices=["dcgan"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4], dest='opts') #입력으로 리스트 받기

parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--norm", default="bnorm", choices=["bnorm", "inorm"], type=str, dest="norm")
#1
parser.add_argument("--network", default="dcgan", choices=["resnet", "unet", "srresnet", "dcgan"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
