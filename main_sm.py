import numpy as np
import gzip
import cPickle as pickle

from keras.utils import np_utils

from drcn import *
from myutils import *


def load_svhn(dataset = '/local/scratch/gif/dataset/SVHN/svhn_gray.pkl.gz'):
    f = gzip.open(dataset,'rb')
    (X_train, y_train), (X_test, y_test) = pickle.load(f)
    f.close()

    idx10 = np.where(y_train == 10)
    y_train[idx10] = 0

    idx10 = np.where(y_test == 10)
    y_test[idx10] = 0
    
    return (X_train.astype('float32'), y_train.astype('uint8')), (X_test.astype('float32'), y_test.astype('uint8'))

def load_mnist32x32(dataset = '/local/scratch/gif/dataset/MNIST/mnist32x32.pkl.gz'):
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set


    train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, 32, 32)
    valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], 1, 32, 32)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, 32, 32)

    return (train_set_x.astype('float32'), train_set_y.astype('uint8')), (valid_set_x.astype('float32'), valid_set_y.astype('uint8')), (test_set_x.astype('float32'), test_set_y.astype('uint8'))


# Load datasets
print('Load datasets')
(Xr_train, y_train), (Xr_test, y_test) = load_svhn() # source
(_, _), (_, _), (Xr_tgt_test, y_tgt_test) = load_mnist32x32() # target

# Convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_tgt_test = np_utils.to_categorical(y_tgt_test, nb_classes)

# Preprocess input images
X_train = preprocess_images(Xr_train, tmin=0, tmax=1)
X_test = preprocess_images(Xr_test, tmin=0, tmax=1)
X_tgt_test = preprocess_images(Xr_tgt_test, tmin=0, tmax=1)

drcn = DRCN()
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

print('Create Model')
drcn.create_model(input_shape=input_shape, dy=nb_classes, nb_filters=[100, 150, 200], kernel_size=(5, 5), pool_size=(2, 2), 
		dropout=0.5, output_activation='softmax')

# print('Train...')
# PARAMDIR = ''
# CONF = 'svhn-mnist_convnet'
# drcn.fit_convnet(X_train, Y_train, 
# 	validation_data=(X_test, Y_test),
# 	test_data=(X_tgt_test, Y_tgt_test),
#  	PARAMDIR=PARAMDIR, CONF=CONF
# )

print('Train convae...')
PARAMDIR = ''
CONF = 'svhn-mnist_drcn'
# drcn.fit_convae(X_tgt_test, validation_data=X_tgt_test[:100], test_data=X_train[:100], PARAMDIR=PARAMDIR, CONF=CONF)
drcn.fit_drcn(X_train, Y_train, X_tgt_test, validation_data=(X_test, Y_test), 
		test_data=(X_tgt_test, Y_tgt_test),
		PARAMDIR=PARAMDIR, CONF=CONF
)

