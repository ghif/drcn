"""
Run DRCN on SVHN (source) --> MNIST (target)

Author: Muhammad Ghifary (mghifary@gmail.com)

"""

import numpy as np
import gzip
import cPickle as pickle

from keras.utils import np_utils

from drcn import *
from myutils import *
from dataset import *

# Load datasets
print('Load datasets')
(Xr_train, y_train), (Xr_test, y_test) = load_svhn(dataset='/local/scratch/gif/dataset/SVHN/svhn_gray.pkl.gz') # source
(_, _), (_, _), (Xr_tgt_test, y_tgt_test) = load_mnist32x32(dataset='/local/scratch/gif/dataset/MNIST/mnist32x32.pkl.gz') # target

# Convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_tgt_test = np_utils.to_categorical(y_tgt_test, nb_classes)

# Preprocess input images
X_train = preprocess_images(Xr_train, tmin=0, tmax=1)
X_test = preprocess_images(Xr_test, tmin=0, tmax=1)
X_tgt_test = preprocess_images(Xr_tgt_test, tmin=0, tmax=1)

print('Create Model')
drcn = DRCN()

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
drcn.create_model(input_shape=input_shape, dense_dim=1024, dy=nb_classes, nb_filters=[100, 150, 200], kernel_size=(3, 3), pool_size=(2, 2), 
		dropout=0.5, bn=False, output_activation='softmax', opt='adam')

print('Train drcn...')
PARAMDIR = ''
CONF = 'svhn-mnist_drcn_v2'
drcn.fit_drcn(X_train, Y_train, X_tgt_test, validation_data=(X_test, Y_test), 
		test_data=(X_tgt_test, Y_tgt_test),
		nb_epoch=50, batch_size=128,
		PARAMDIR=PARAMDIR, CONF=CONF
)
