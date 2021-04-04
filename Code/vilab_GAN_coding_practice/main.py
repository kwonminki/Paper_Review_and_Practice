import argparse

import os
from pathlib import Path

from train import *
from test import test

#PATH
HOME_PATH = Path("/home/mingi/mingi")
DATA_PATH = Path("/datasets")
PATH = HOME_PATH/"vilab_GAN_coding_practice"

#Parser
parser = argparse.ArgumentParser(description="DCGAN",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--epochs", default=10, type=int, dest="epochs")

parser.add_argument("--data_dir", default=DATA_PATH/"CELEBA/img_align_celeba", type=str, dest="data_dir")
parser.add_argument("--project_dir", default=PATH, type=str, dest="project_dir")


parser.add_argument("--device", default='cuda', type=str, dest="device")

parser.add_argument("--mode", default="train", choices=["train", "test", "val"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--task", default=None, type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=[], dest='opts') #입력으로 리스트 받기

parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--norm", default="bnorm", choices=["bnorm", "inorm"], type=str, dest="norm")
#1
parser.add_argument("--network", default="dcgan", choices=["resnet", "dcgan", "stylegan", "stylegan2"], type=str, dest="network")
parser.add_argument("--advloss", default="vanilla", choices=["vanilla", "wasserstein", "ls", "hinge"], type=str, dest="advloss")
# parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")
parser.add_argument("--foldername", default='mingi', type=str, dest="foldername")

parser.add_argument("--sample_every", default=1, type=int, dest="sample_every")
parser.add_argument("--chpt_every", default=1, type=int, dest="chpt_every")
parser.add_argument("--eval_every", default=10, type=int, dest="eval_every")
parser.add_argument("--save_image_num", default=5, type=int, dest="save_image_num")
parser.add_argument("--print_every", default=10, type=int, dest="print_every")

args = parser.parse_args()

if __name__ == "__main__":

    if args.mode == "train":
        if args.save_image_num > args.batch_size:
            print("save_image_num is more than batch_size")
            exit(-1)
        train(args)
    elif args.mode == "test":
        test(args)
