from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

sys.path.insert(0, '../')
from train import data_augmentation

_DATA_AUMENTATION_ENABLED = True
_N_AUGMENTATIONS = 30

_NPY_DATASET_DIR = '../data/'
_CONVERTED_DATASET_DIR = '/tmp/lfw17-augmented'
_IMG_DATASET_DIR = _CONVERTED_DATASET_DIR + '/faces_photos'

_MODE = 'TRAIN'
# _MODE = 'TEST'

_IMGS_W = 37
_IMGS_H = 50
_IMGS_WH = (_IMGS_W, _IMGS_H)
_IMGS_HW = (_IMGS_H, _IMGS_W) # actually used for reshape

_DESIRED_DIM = 256

def _create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def _plot_img(i):
    plt.gray() # only to see the images
    img = np.reshape(X[i], _IMGS_HW)
    print(img.shape)
    imgplot = plt.imshow(img)
    plt.show()

def _resize_to_desired_keep_ar(img, desired_dim):
    if (img.width > desired_dim and img.height > desired_dim):
        return img
    else:
        max_ratio_for_desired_size = max(desired_dim / img.width, desired_dim / img.height)
        img_dims = (int(max_ratio_for_desired_size * img.width), int(max_ratio_for_desired_size * img.height))
        img = img.resize(img_dims)
        return img

if (_MODE == 'TRAIN'):
    # load npy data
    X = np.load(_NPY_DATASET_DIR + 'X_train.npy')
    y = np.load(_NPY_DATASET_DIR + 'y_train.npy')

    X, y = data_augmentation(X,
                             y,
                             n_augmentations_per_image=_N_AUGMENTATIONS,
                             max_rotation_angle=25,
                             horizontal_flip_chance=0.5,
                             rotation_chance=1 - (1.0 / _N_AUGMENTATIONS))

    _NUM_FACES = y.shape[0]

    print('Dataset size: %d' % (_NUM_FACES))

    for i in range(_NUM_FACES):
        label = y[i]
        img_destination_folder = _IMG_DATASET_DIR + '/' + str(label)
        img_destination_path = img_destination_folder + '/' + str(i) + '.jpg'
        _create_dir_if_not_exists(img_destination_folder)
        im = np.reshape(X[i], _IMGS_HW)
        im = Image.fromarray(im)
        im = im.convert('RGB')
        im = _resize_to_desired_keep_ar(im, _DESIRED_DIM)
        im.save(img_destination_path)

elif (_MODE == 'TEST'):
    # load npy data
    X = np.load(_NPY_DATASET_DIR + 'X_test.npy')
    _NUM_FACES = X.shape[0]

    for i in range(_NUM_FACES):
        img_destination_folder = _IMG_DATASET_DIR + '-augmented/'
        img_destination_path = img_destination_folder + '/' + str(i) + '.jpg'
        _create_dir_if_not_exists(img_destination_folder)
        im = np.reshape(X[i], _IMGS_HW)
        im = Image.fromarray(im)
        im = im.convert('RGB')
        im = _resize_to_desired_keep_ar(im, _DESIRED_DIM)
        im.save(img_destination_path)

else:
    print('\nUnknown mode')
    sys.exit(0)