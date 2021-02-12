import os
import numpy as np

import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        list_data = os.listdir(self.data_dir)

        list_label = [f for f in list_data if f.startswith('label')]
        list_input = [f for f in list_data if f.startswith('input')]

        list_label.sort()
        list_input.sort()

        self.list_input = list_input
        self.list_label = list_label

    def __len__(self):
        return len(self.list_label)

    def __getitem__(self, index):
        label = np.load(self.data_dir/self.list_label[index])
        input = np.load(self.data_dir/self.list_input[index])

        label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        ##numpy는 Y,X,CH 순서인데 tensor는 CH,Y,X 순서이다. 그래서 바꿔준다.
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# 노말라이제이션 구현
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std
        #label은 0,1로 이루어져 있으므로 Normalization하면 안됨.

        data = {'label': label, 'input': input}

        return data

#Flip
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        #좌우
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        #위아래
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
