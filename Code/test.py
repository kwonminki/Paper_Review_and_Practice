import argparse
import yaml
import multiprocessing

import numpy as np
import tensorflow as tf

from trainer3 import *
from logger import *
# from dataset2 import *
from dataloader2 import *





if __name__=='__main__':
    np.random.seed(10)
    # parse options
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True, help='the name of current job')
    parser.add_argument("--config", type=str, default='./configs/CUB.yaml', help='path to config')
    parser.add_argument('--start_epoch', type=int, default=0, help='resume training from this epoch')
    parser.add_argument("--base", action='store_true', help='whether use base class')
    parser.add_argument('--support', type=int, default=5, help='resume training from this epoch')
#2021수정
    parser.add_argument('--all', action='store_true', help='test')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    # load dataset
    data_path = config['dataset_params']['dataset_path']
    batch_size = config['dataset_params']['batch_size']
    if config['dataset_params']['data_type'] =='CUB':
        #val_dataset = ImageLoader(data_path, train=False, num_classes=200, novel_only=True)
#2021수정
        if args.all:
            val_dataset = ImageLoader(data_path, train=False, num_classes=200 , novel_only=False)
        else:
            val_dataset = ImageLoader(data_path, train=False, num_classes=200 - 100*(args.base), novel_only=(args.base == False))          

        val_generator = val_dataset.get_dataset(batch_size=batch_size, shuffle=False, drop_last=False)
    # else:    
    #     multiprocessing.set_start_method('spawn')
    #     dataset = SimpleDataset('train', config['dataset_params']['dataset_path'])
    #     dataset_generator = dataset.get_data_generator(image_size=config['dataset_params']['image_size'],
    #                                 batch_size=config['dataset_params']['batch_size'], repeat_num=1, aug=config['dataset_params']['aug'])
    #     iter_per_epoch = len(dataset_generator)

    # create trainer
    if args.base:
        trainer = Trainer(config['model_params'], args.job_name, pretrain=False, support=args.support)
    else:
        trainer = Trainer(config['model_params'], args.job_name, pretrain=False, support=args.support)
    
    trainer.load_checkpoint(args.start_epoch) #0이면 imprinting

    # validation
    trainer.validate(val_generator)
