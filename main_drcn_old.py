import lasagne
from lasagne.layers import get_output, get_output_shape
from lasagne.layers import InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, FlattenLayer, DropoutLayer
from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow

try:
	from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayerFast
	from lasagne.layers.dnn import Pool2DDNNLayer as MaxPool2DLayerFast
	print('Usage: CuDNN (fast)')

except ImportError:
	from lasagne.layers import Conv2DLayer as Conv2DLayerFast
	from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
	print('Usage: Lasagne (slow)')	

from lasagne.nonlinearities import rectify, sigmoid, softmax
from lasagne.updates import rmsprop
from lasagne.objectives import categorical_crossentropy, squared_error, binary_crossentropy
from lasagne.init import GlorotUniform


from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils
import theano
import theano.tensor as T

import numpy as np
import sys
import gzip
from time import strftime
import cPickle as pickle
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from visualizer import *
from loader import *
from myutils import *

from sklearn import preprocessing


# hyperparameters
confnet = {
	'lr': 1e-3,
	'batch_size': 128,
	'test_batch_size': 1024,
	'nb_epoch': 100,
	'is_aug': 0,
	'dropout_rate': 0.5
}

confae = {
	'lr': 1e-3,
	'batch_size': 128,
	'test_batch_size': 1024,
	'nb_epoch': 100,
	'is_aug': 0,
	'dropout_rate': 0.0
}


src = 'mnist'
tgt = 'svhn'

RESFILE = 'results/'+src+'-'+tgt+'_drcn_results_drop%.1f_aug%d.pkl.gz' % (confnet['dropout_rate'], confnet['is_aug'])
PARAMFILE = 'results/'+src+'-'+tgt+'_drcn_weights_drop%.1f_aug%d.pkl.gz' % (confnet['dropout_rate'], confnet['is_aug'])
print(RESFILE)


# Load data
if src == 'svhn':
	(X_train, Y_train), (X_test, Y_test) = load_svhn()
	(_, _), (_, _), (X_tgt_test, Y_tgt_Test) = load_mnist32x32()
elif src == 'mnist':
	(X_train, Y_train), (_, _), (X_test, Y_test) = load_mnist32x32()
	(_, _), (X_tgt_test, Y_tgt_test) = load_svhn()


print('Preprocess data ...')
X_train = min_max(X_train)
X_test = min_max(X_test)
X_tgt_test = min_max(X_tgt_test)

[n, c, d1, d2] = X_train.shape



###### CONVNET ######
print('Create ConvNet....')
Xnet_ = T.ftensor4('x')
Ynet_ = T.ivector('y')

lnet_in = InputLayer((None, c, d1, d2), Xnet_)
lnet_conv1 = Conv2DLayerFast(lnet_in, 100, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
lnet_pool1 = MaxPool2DLayerFast(lnet_conv1, (2,2))

lnet_conv2 = Conv2DLayerFast(lnet_pool1, 150, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
lnet_pool2 = MaxPool2DLayerFast(lnet_conv2, (2,2))

lnet_conv3 = Conv2DLayerFast(lnet_pool2, 200, (3,3), W=GlorotUniform(), nonlinearity=rectify)
lnet_conv3_flat = FlattenLayer(lnet_conv3)

lnet_dense4 = DenseLayer(lnet_conv3_flat, 300, W=GlorotUniform(), nonlinearity=rectify)
lnet_dense4_drop = DropoutLayer(lnet_dense4, p=confnet['dropout_rate'])

convnet = DenseLayer(lnet_dense4_drop, 10, nonlinearity=softmax)

print('[ConvNet] define loss, optimizer, and compile')
Ynet_train_pred_ = get_output(convnet)
loss_ = categorical_crossentropy(Ynet_train_pred_, Ynet_)
loss_ = loss_.mean()
acc_ = T.mean(T.eq(T.argmax(Ynet_train_pred_, axis=1), Ynet_), dtype=theano.config.floatX)

params_ = lasagne.layers.get_all_params(convnet, trainable=True)
updates_ = rmsprop(loss_, params_, learning_rate=confnet['lr'])
train_net_fn = theano.function([Xnet_, Ynet_], [loss_, acc_], updates=updates_)

# test loss
Ynet_test_pred_ = get_output(convnet, deterministic=True)
test_net_loss_ = categorical_crossentropy(Ynet_test_pred_, Ynet_)
test_net_loss_ = test_net_loss_.mean()

# test accuracy
test_net_acc_ = T.mean(T.eq(T.argmax(Ynet_test_pred_, axis=1), Ynet_), dtype=theano.config.floatX)
test_net_fn = theano.function([Xnet_, Ynet_], [test_net_loss_, test_net_acc_])

###############


##### CONVEA ####
print('Create ConvAE....')
Xae_ = T.ftensor4('x')
Yae_ = T.ftensor4('y')

lae_in = InputLayer((None, c, d1, d2), Xae_)
print(get_output_shape(lae_in))
lae_conv1 = Conv2DLayerFast(lae_in, 100, (5,5), pad=(2,2), W=lnet_conv1.W, nonlinearity=rectify)
print(get_output_shape(lae_conv1))
lae_pool1 = MaxPool2DLayerFast(lae_conv1, (2,2))
print(get_output_shape(lae_pool1))

lae_conv2 = Conv2DLayerFast(lae_pool1, 150, (5,5), pad=(2,2), W=lnet_conv2.W, nonlinearity=rectify)
print(get_output_shape(lae_conv2))
lae_pool2 = MaxPool2DLayerFast(lae_conv2, (2,2))
print(get_output_shape(lae_pool2))

lae_conv3 = Conv2DLayerFast(lae_pool2, 200, (3,3), W=lnet_conv3.W, nonlinearity=rectify)
print(get_output_shape(lae_conv3))
[nd, nf, dc1, dc2] = get_output_shape(lae_conv3)

lae_conv3_flat = FlattenLayer(lae_conv3)
print(get_output_shape(lae_conv3_flat))
[_, dflat] = get_output_shape(lae_conv3_flat)

lae_dense4 = DenseLayer(lae_conv3_flat, 300, W=lnet_dense4.W, nonlinearity=rectify)
print(get_output_shape(lae_dense4))

lae_dense5 = DenseLayer(lae_dense4, dflat, W=GlorotUniform(), nonlinearity=rectify)
print(get_output_shape(lae_dense5))


lae_dense5_reshape = ReshapeLayer(lae_dense5, ([0], nf, dc1, -1)) # lae_conv3
print(get_output_shape(lae_dense5_reshape))

lae_deconv6 = Conv2DLayerFast(lae_dense5_reshape, 150, (3,3), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
print(get_output_shape(lae_deconv6))
lae_unpool6 = Upscale2DLayer(lae_deconv6, (2,2))
print(get_output_shape(lae_unpool6))

lae_deconv7 = Conv2DLayerFast(lae_unpool6, 100, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
print(get_output_shape(lae_deconv7))
lae_unpool7 = Upscale2DLayer(lae_deconv7, (2,2))
print(get_output_shape(lae_unpool7))

convae = Conv2DLayerFast(lae_unpool7, 1, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=sigmoid)
print(get_output_shape(convae))



print('[ConvAE] define loss, optimizer, and compile')
Yae_pred_ = get_output(convae)
loss_ = binary_crossentropy(Yae_pred_, Yae_)
loss_ = loss_.mean()

params_ = lasagne.layers.get_all_params(convae, trainable=True)
updates_ = rmsprop(loss_, params_, learning_rate=confae['lr'])
train_ae_fn = theano.function([Xae_, Yae_], loss_, updates=updates_)
pred_ae_fn = theano.function([Xae_], Yae_pred_)

##################

if confnet['is_aug']:
    ddatagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width))
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=False, # randomly flip images
        vertical_flip=False
    ) # randomly flip images
else:
    ddatagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=0., # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0., # randomly shift images horizontally (fraction of total width)
        height_shift_range=0., # randomly shift images vertically (fraction of total height)
        horizontal_flip=False, # randomly flip images
        vertical_flip=False
    ) # randomly flip images

gdatagen = ImageDataGenerator(
    featurewise_center=False, # set input mean to 0 over the dataset
    samplewise_center=False, # set each sample mean to 0
    featurewise_std_normalization=False, # divide inputs by std of the dataset
    samplewise_std_normalization=False, # divide each input by its std
    zca_whitening=False, # apply ZCA whitening
    rotation_range=0., # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0., # randomly shift images horizontally (fraction of total width)
    height_shift_range=0., # randomly shift images vertically (fraction of total height)
    horizontal_flip=False, # randomly flip images
    vertical_flip=False
) # randomly flip images


print('Train DRCN ....')
losses = []
test_losses = []
tgt_losses = []

accs = []
test_accs = []
tgt_accs = []

elapsed_times = []

total_batches_net = n / confnet['batch_size']

n2 = X_tgt_test.shape[0]
total_batches_ae = n2 / confae['batch_size']



for epoch in range(confnet['nb_epoch']):
	start_time = time.time()

	print('--- ConvAE ---')
	loss_ae = 0
	nbatch = 0
	for X_batch, Y_batch in gdatagen.flow(X_tgt_test, X_tgt_test, batch_size=confae['batch_size']):
		X_batch = X_batch.astype('float32')
		Y_batch = Y_batch.astype('float32')

		loss_ae += train_ae_fn(X_batch, Y_batch)

		nbatch += 1

		if nbatch > total_batches_ae:
			break;

	loss_ae /= nbatch
	print('---> ConvAE loss: ', loss_ae)

	show_filter(pred_ae_fn(X_train[:100,:]), grayscale=True, filename=src+'-'+tgt+'_drcn_X_pred_source.png')
	show_filter(pred_ae_fn(X_tgt_test[:100,:]), grayscale=True, filename=src+'-'+tgt+'_drcn_X_pred_target.png')

	print('--- ConvNet ---')
	nbatch = 0
	loss = 0
	acc = 0
	for X_batch, Y_batch in ddatagen.flow(X_train, Y_train, batch_size=confnet['batch_size']):
		X_batch = X_batch.astype('float32')
		Y_batch = Y_batch.astype('uint8')

		[o1, o2] = train_net_fn(X_batch, Y_batch)
		loss += o1
		acc += o2

		nbatch += 1

		if nbatch > total_batches_net:
			break;

	etime = time.time() - start_time
	elapsed_times.append(etime)

	loss = loss / nbatch
	acc = acc / nbatch

	losses.append(loss)
	accs.append(acc)

	[test_loss, test_acc] = evaluate(test_net_fn, X_test, Y_test, batch_size=confnet['test_batch_size'])
	test_losses.append(test_loss)
	test_accs.append(test_acc)

	[tgt_loss, tgt_acc] = evaluate(test_net_fn, X_tgt_test, Y_tgt_test, batch_size=confnet['test_batch_size'])
	tgt_losses.append(tgt_loss)
	tgt_accs.append(tgt_acc)

	print('Epoch %d: Training (Loss: %.4f, Acc: %.4f), Test (Loss: %.4f, Acc: %.4f), Target (Loss: %.4f, Acc: %.4f), Time: %.2f seconds'
		% ((epoch+1), loss, acc, test_loss, test_acc, tgt_loss, tgt_acc, etime))

	res = {
		'losses': losses,
		'accs': accs,
		'test_losses': test_losses,
		'test_accs': test_accs,
		'tgt_losses': tgt_losses,
		'tgt_accs': tgt_accs
	}

	pickle.dump((res, confnet, confae), gzip.open(RESFILE,'wb'))

	# store params
	params_net = lasagne.layers.get_all_param_values(convnet)
	params_ae = lasagne.layers.get_all_param_values(convae)
	pickle.dump((params_net, params_ae), gzip.open(PARAMFILE, 'wb'))

