import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from PIL import Image
from util import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None): #task와 opts도 받아온다.
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        # self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')] #특정 확장자만 받아온다.

        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        #png파일을 load 
        img = Image.open(os.path.join(self.data_dir, self.lst_data[index])).convert('RGB')

        if self.transform:
            img = self.transform(img)

        data = {'label': img}

        return data


class ToTensor(object):
    def __call__(self, data):
        ##numpy는 Y,X,CH 순서인데 tensor는 CH,Y,X 순서이다. 그래서 바꿔준다.
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

# 노말라이제이션 구현
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']

        # input = (input - self.mean) / self.std
        # #label은 0,1로 이루어져 있으므로 Normalization하면 안됨. - segmentation에서는 그랬는데 
        # #리그레션은 아님.
        # label = (label - self.mean) / self.std

        # data = {'label': label, 'input': input}

        # return data
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data

#Flip
class RandomFlip(object):
    # def __call__(self, data):
    #     label, input = data['label'], data['input']

    #     #좌우
    #     if np.random.rand() > 0.5:
    #         label = np.fliplr(label)
    #         input = np.fliplr(input)
    #     #위아래
    #     if np.random.rand() > 0.5:
    #         label = np.flipud(label)
    #         input = np.flipud(input)

    #     data = {'label': label, 'input': input}

    #     return data
    def __call__(self, data):

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        return data


#Crop
class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1],
                                                    self.shape[2]))

        return data