#%% 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path

#PATH
HOME_PATH = Path("/home/mingi/mingi")
DATA_PATH = HOME_PATH/"data"
PATH = HOME_PATH/"UNet"
dir_data = DATA_PATH/'ISBI_EM_stacks'
#데이터 불러오기
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(dir_data/name_label)
img_input = Image.open(dir_data/name_input)

ny, nx = img_label.size
nframe = img_label.n_frames

#개수 설정 및 디렉토리 생성
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = dir_data/'train'
dir_save_val = dir_data/'val'
dir_save_test = dir_data/'test'

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

#랜덤하게
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

#save_train
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

#save_val
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

#save_test
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

#%%
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()

# %%
