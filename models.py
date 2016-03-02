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

from lasagne.nonlinearities import rectify, sigmoid, softmax, linear
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


from visualizer import *
from myutils import *

def select_loss(loss_str='squared_error'):
	if loss_str == 'squared_error':
		return squared_error
	elif loss_str == 'binary_crossentropy':
		return binary_crossentropy
	elif loss_str == 'categorical_crossentropy':
		return categorical_crossentropy

class ConvAE:
	def __init__(self):
		self.X_ = T.ftensor4('x')
		self.Y_ = T.ftensor4('y')

	def create_architecture(self, input_shape, dense_dim=1024, input_var_=None, output_var_=None, convnet_=None, 
		is_enc_fixed=False):
		

		print('[ConvAE: create_architecture]')
		if input_var_ is not None:
			self.X_ = input_var_

		if output_var_ is not None:
			self.Y_ = output_var_


		(c, d1, d2) = input_shape

		self.lin = InputLayer((None, c, d1, d2), self.X_)
		if convnet_ is not None:
			self.lconv1 = Conv2DLayerFast(self.lin, 100, (5,5), pad=(2,2), W=convnet_.lconv1.W, nonlinearity=rectify)
		else:
			self.lconv1 = Conv2DLayerFast(self.lin, 100, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)

		self.lpool1 = MaxPool2DLayerFast(self.lconv1, (2,2))

		if convnet_ is not None:
			self.lconv2 = Conv2DLayerFast(self.lpool1, 150, (5,5), pad=(2,2), W=convnet_.lconv2.W, nonlinearity=rectify)
		else:
			self.lconv2 = Conv2DLayerFast(self.lpool1, 150, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)

		self.lpool2 = MaxPool2DLayerFast(self.lconv2, (2,2))

		if convnet_ is not None:
			self.lconv3 = Conv2DLayerFast(self.lpool2, 200, (3,3), W=convnet_.lconv3.W, nonlinearity=rectify)
		else:
			self.lconv3 = Conv2DLayerFast(self.lpool2, 200, (3,3), W=GlorotUniform(), nonlinearity=rectify)
		[nd, nf, dc1, dc2] = get_output_shape(self.lconv3)

		self.lconv3_flat = FlattenLayer(self.lconv3)
		[_, dflat] = get_output_shape(self.lconv3_flat)

		if convnet_ is not None:
			self.ldense1 = DenseLayer(self.lconv3_flat, dense_dim, W=convnet_.ldense1.W, nonlinearity=rectify)
		else:
			self.ldense1 = DenseLayer(self.lconv3_flat, dense_dim, W=GlorotUniform(), nonlinearity=rectify)

		if convnet_ is not None:
			self.ldense2 = DenseLayer(self.ldense1, dense_dim, W=convnet_.ldense2.W, nonlinearity=rectify)
		else:
			self.ldense2 = DenseLayer(self.ldense1, dense_dim, W=GlorotUniform(), nonlinearity=rectify)

		self.ldense3 = DenseLayer(self.ldense2, dflat, W=GlorotUniform(), nonlinearity=rectify)
		self.ldense3_reshape = ReshapeLayer(self.ldense3, ([0], nf, dc1, -1)) # lae_conv3

		self.ldeconv1 = Conv2DLayerFast(self.ldense3_reshape, 150, (3,3), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
		self.lunpool1 = Upscale2DLayer(self.ldeconv1, (2,2))

		self.ldeconv2 = Conv2DLayerFast(self.lunpool1, 100, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
		self.lunpool2 = Upscale2DLayer(self.ldeconv2, (2,2))

		self.model_ = Conv2DLayerFast(self.lunpool2, 1, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=linear)


		self.is_enc_fixed = is_enc_fixed


	def compile(self, lr=1e-4, loss_function='squared_error'):
		self.lr = lr
		print('[ConvAE: compile]')
		self.loss_function = select_loss(loss_function)

		Y_pred_ = get_output(self.model_)
		self.loss_ = self.loss_function(Y_pred_, self.Y_).mean()
		
		params_ = lasagne.layers.get_all_params(self.model_, trainable=True)
		if self.is_enc_fixed:
			updates_ = rmsprop(self.loss_, params_[10:len(params_)], learning_rate=self.lr)
		else:
			updates_ = rmsprop(self.loss_, params_, learning_rate=self.lr)
		self.train_fn = theano.function([self.X_, self.Y_], self.loss_, updates=updates_)

		Y_test_pred_ = get_output(self.model_, deterministic=True)
		self.pred_fn = theano.function([self.X_], Y_test_pred_)
		
	def predict(self, X):
		return self.pred_fn(X)


	def train(self, X, batch_size=128, nb_epoch=50, denoising=False, 
			X_valid = None,
			RESFILE=None, PARAMFILE=None, PREDICTPREFIX=None):

		print('[ConvAE: train]')
		datagen = ImageDataGenerator(
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


		self.losses = []
		self.elapsed_times = []

		n = X.shape[0]
		total_batches = n / batch_size

		Y = np.copy(X)

		for epoch in range(nb_epoch):
			loss = 0
			nbatch = 0
			start_time = time.time()

			for X_batch, Y_batch in datagen.flow(X, Y, batch_size=batch_size):
				X_batch = X_batch.astype('float32')
				Y_batch = Y_batch.astype('float32')

				loss += self.train_fn(X_batch, Y_batch)
				nbatch += 1
				if nbatch > total_batches:
					break;

			loss /= nbatch
			etime = time.time() - start_time

			self.elapsed_times.append(etime)
			self.losses.append(loss)
			
			print('Epoch %d: Trainig (Loss : %.4f), Time : %.2f' % ((epoch+1), loss, etime))

			if PREDICTPREFIX is not None:
				show_filter(X[:100], grayscale=True, filename=PREDICTPREFIX+'X-tgt-orig.png')
				show_filter(self.pred_fn(X[:100]), grayscale=True, filename=PREDICTPREFIX+'X-tgt-pred.png')
				if X_valid is not None:
					show_filter(X_valid[:100], grayscale=True, filename=PREDICTPREFIX+'X-src-orig.png')
					show_filter(self.pred_fn(X_valid[:100]), grayscale=True, filename=PREDICTPREFIX+'X-src-pred.png')


			self.res = {
				'losses':self.losses,
				'elapsed_times':self.elapsed_times,
				'lr': self.lr,
				'batch_size':batch_size,
				'nb_epoch':nb_epoch
			}

			if RESFILE is not None:
				pickle.dump(self.res, gzip.open(RESFILE,'wb'))

		if PARAMFILE is not None:
			params = lasagne.layers.get_all_param_values(self.model_)
			pickle.dump(params, gzip.open(PARAMFILE, 'wb'))

	def load_weights(self, PARAMFILE):
		params = pickle.load(gzip.open(PARAMFILE, 'rb'))
		lasagne.layers.set_all_param_values(self.model_, params)
	


class ConvNet:
	def __init__(self):
		self.X_ = T.ftensor4('x')
		self.Y_ = T.ivector('y')

	def create_architecture(self, input_shape, dense_dim=1024, dout=10, dropout=0.5, 
		input_var_=None, output_var_=None, enc_weights=None):
		
		print('[ConvNet: create_architecture] dense_dim:', dense_dim)

		if input_var_ is not None:
			self.X_ = input_var_

		if output_var_ is not None:
			self.Y_ = output_var_

		self.dropout = dropout
		(c, d1, d2) = input_shape

		self.lin = InputLayer((None, c, d1, d2), self.X_)
		self.lconv1 = Conv2DLayerFast(self.lin, 100, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
		self.lpool1 = MaxPool2DLayerFast(self.lconv1, (2,2))

		self.lconv2 = Conv2DLayerFast(self.lpool1, 150, (5,5), pad=(2,2), W=GlorotUniform(), nonlinearity=rectify)
		self.lpool2 = MaxPool2DLayerFast(self.lconv2, (2,2))

		self.lconv3 = Conv2DLayerFast(self.lpool2, 200, (3,3), W=GlorotUniform(), nonlinearity=rectify)		
		self.lconv3_flat = FlattenLayer(self.lconv3)

		self.ldense1 = DenseLayer(self.lconv3_flat, dense_dim, W=GlorotUniform(), nonlinearity=rectify)
		self.ldense1_drop = self.ldense1
		if dropout > 0:
			self.ldense1_drop = DropoutLayer(self.ldense1, p=dropout)

		self.ldense2 = DenseLayer(self.ldense1_drop, dense_dim, W=GlorotUniform(), nonlinearity=rectify)
		self.ldense2_drop = self.ldense2
		if dropout > 0:
			self.ldense2_drop = DropoutLayer(self.ldense2_drop, p=dropout)

		self.model_ = DenseLayer(self.ldense2_drop, dout, W=GlorotUniform(), nonlinearity=softmax)

		self.enc_weights = enc_weights
		if enc_weights is not None:
			lasagne.layers.set_all_param_values(self.model_, enc_weights)
	
	def compile(self, lr=1e-4, loss_function='categorical_crossentropy'):
		self.lr = lr
		print('[ConvNet: compile]')
		self.loss_function = select_loss(loss_function)


		Y_pred_ = get_output(self.model_)
		self.loss_ = self.loss_function(Y_pred_, self.Y_).mean()
		self.acc_ = T.mean(T.eq(T.argmax(Y_pred_, axis=1), self.Y_), dtype=theano.config.floatX)
		params_ = lasagne.layers.get_all_params(self.model_, trainable=True)
		updates_ = rmsprop(self.loss_, params_, learning_rate=self.lr)
		self.train_fn = theano.function([self.X_, self.Y_], [self.loss_, self.acc_], updates=updates_)

		Y_test_pred_ = get_output(self.model_, deterministic=True)
		self.test_loss_ = self.loss_function(Y_test_pred_, self.Y_).mean()
		self.test_acc_ = T.mean(T.eq(T.argmax(Y_test_pred_, axis=1), self.Y_), dtype=theano.config.floatX)
		self.pred_fn = theano.function([self.X_], Y_test_pred_)
		self.test_fn = theano.function([self.X_, self.Y_], [self.test_loss_, self.test_acc_])

	def evaluate(self, X, Y, batch_size=512):
		loss = 0
		acc = 0
		nbatch = 0
		for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size):
			X = X.astype('float32')
			Y = Y.astype('uint8')
			[o1, o2] = self.test_fn(X_batch, Y_batch)
			loss += o1
			acc += o2
			nbatch += 1
		loss /= nbatch
		acc /= nbatch
		return loss, acc

	def predict(self, X):
		return self.pred_fn(X)

	def train(self, X, Y, batch_size=128, nb_epoch=50, shuffle=False, augmentation=False, 
				X_test = None, Y_test = None, X_tgt=None, Y_tgt=None,
				RESFILE=None, PARAMFILE=None):

		self.augmentation = augmentation
		print('[ConvNet: train]')
		if augmentation:
			datagen = ImageDataGenerator(
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
			datagen = ImageDataGenerator(
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


		self.losses = []
		self.test_losses = []
		self.tgt_losses = []

		self.accs = []
		self.test_accs = []
		self.tgt_accs = []

		self.elapsed_times = []

		n = X.shape[0]

		total_batches = n / batch_size

		
		for epoch in range(nb_epoch):
			loss = 0
			acc = 0
			nbatch = 0
			start_time = time.time()

			for X_batch, Y_batch in datagen.flow(X, Y, batch_size=batch_size, shuffle=shuffle):
				X_batch = X_batch.astype('float32')
				Y_batch = Y_batch.astype('uint8')

				[o1, o2] = self.train_fn(X_batch, Y_batch)
				loss += o1
				acc += o2

				nbatch += 1

				if nbatch > total_batches:
					break;

			loss /= nbatch
			acc /= nbatch
			etime = time.time() - start_time

			self.elapsed_times.append(etime)
			self.losses.append(loss)
			self.accs.append(acc)


			# Evaluate
			[test_loss, test_acc] = self.evaluate(X_test, Y_test, batch_size=512)
			self.test_losses.append(test_loss)
			self.test_accs.append(test_acc)

			[tgt_loss, tgt_acc] = self.evaluate(X_tgt, Y_tgt, batch_size=512)
			self.tgt_losses.append(tgt_loss)
			self.tgt_accs.append(tgt_acc)

			
			print('Epoch %d: Training (Loss: %.4f, Acc: %.4f), Test (Loss: %.4f, Acc: %.4f), Target (Loss: %.4f, Acc: %.4f), Time: %.2f seconds'
				% ((epoch+1), loss, acc, test_loss, test_acc, tgt_loss, tgt_acc, etime))



			self.res = {
				'losses':self.losses,
				'accs': self.accs,
				'test_losses': self.test_losses,
				'test_accs': self.test_accs,
				'tgt_losses': self.tgt_losses,
				'tgt_accs': self.tgt_accs,
				'elapsed_times':self.elapsed_times,
				'lr': self.lr,
				'dropout':self.dropout,
				'augmentation':self.augmentation,
				'batch_size':batch_size,
				'nb_epoch':nb_epoch
			}

			if RESFILE is not None:
				pickle.dump(self.res, gzip.open(RESFILE,'wb'))

		if PARAMFILE is not None:
			params = lasagne.layers.get_all_param_values(self.model_)
			pickle.dump(params, gzip.open(PARAMFILE, 'wb'))


class DRCN:
	def __init__(self, input_shape, output_dim, net_config, ae_config, enc_weights=None):
		self.X_ = T.ftensor4('x')
		self.Xout_ = T.ftensor4('x_out')
		self.Yout_ = T.ivector('y_out')
		self.convnet_ = ConvNet()
		self.convae_ = ConvAE()

		self.net_config = net_config
		self.ae_config = ae_config

		self.enc_weights = enc_weights
		self.is_enc_fixed = False
		if self.enc_weights is not None:
			self.is_enc_fixed = True

		self.create_architecture(input_shape, dense_dim=net_config['dense_dim'], 
			dout=output_dim, dropout=net_config['dropout'])
		self.compile(lr_net=net_config['lr'], lr_ae=ae_config['lr'])

	def create_architecture(self, input_shape, dense_dim=1024, dout=10, dropout=0.5):
		self.convnet_.create_architecture(input_shape, dout=dout, dense_dim=dense_dim, dropout=dropout, 
			input_var_=self.X_, output_var_=self.Yout_, enc_weights=self.enc_weights)

		
		self.convae_.create_architecture(input_shape, input_var_=self.X_, dense_dim=dense_dim,
			output_var_=self.Xout_, convnet_=self.convnet_, is_enc_fixed=self.is_enc_fixed)

	def compile(self, lr_net=1e-4, lr_ae=1e-4):
		self.lr_net = lr_net
		self.lr_ae = lr_ae
		self.convnet_.compile(lr=lr_net, loss_function=self.net_config['loss'])
		self.convae_.compile(lr=lr_ae, loss_function=self.ae_config['loss'])


	def train(self, X, Y, 
			X_test=None, Y_test=None, X_tgt=None, Y_tgt=None,
			RESFILE=None, PARAMFILE=None, PREDICTPREFIX=None):

		print('[DRCN: train]')

		if self.net_config['augmentation']:
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
		else:
			ddatagen = ImageDataGenerator(
			    featurewise_center=False, # set input mean to 0 over the dataset
			    samplewise_center=False, # set each sample mean to 0
			    featurewise_std_normalization=False, # divide inputs by std of the dataset
			    samplewise_std_normalization=False, # divide each input by its std
			    zca_whitening=False, # apply ZCA whitening
			    rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
			    width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
			    height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
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

		self.losses_ae = []
		self.losses = []
		self.test_losses = []
		self.tgt_losses = []

		self.accs = []
		self.test_accs = []
		self.tgt_accs = []

		self.elapsed_times = []

		

		total_batches = X.shape[0] / self.net_config['batch_size']

		total_batches_ae = X_tgt.shape[0] / self.ae_config['batch_size']

		if self.ae_config['input'] == 's':
			X_ae = np.copy(X)
		elif self.ae_config['input'] == 't':
			X_ae = np.copy(X_tgt)
		elif self.ae_config['input'] == 'st':
			X_ae = np.concatenate([X, X_tgt])

		
		for epoch in range(self.net_config['nb_epoch']):

			# ========== CONVAE =======
			start_time = time.time()
			loss_ae = 0
			nbatch = 0

			# print('AE training : ',X_ae.shape)
			for X_batch, Y_batch in gdatagen.flow(X_ae, np.copy(X_ae), batch_size=self.ae_config['batch_size'], shuffle=self.ae_config['shuffle']):
				if self.ae_config['denoising'] > 0.:
					X_batch = get_corrupted_output(X_batch, corruption_level=self.ae_config['denoising']).astype('float32')
				else:
					X_batch = X_batch.astype('float32')

				Y_batch = Y_batch.astype('float32')

				loss_ae += self.convae_.train_fn(X_batch, Y_batch)
				nbatch += 1

				if nbatch > total_batches_ae:
					break;

			loss_ae /= nbatch
			self.losses_ae.append(loss_ae)

			# ============= CONVNET +==============
			loss = 0
			acc = 0
			nbatch = 0
			
			for X_batch, Y_batch in ddatagen.flow(X, Y, batch_size=self.net_config['batch_size'], shuffle=self.net_config['shuffle']):
				X_batch = X_batch.astype('float32')
				Y_batch = Y_batch.astype('uint8')

				if self.is_enc_fixed:
					[o1, o2] = self.convnet_.test_fn(X_batch, Y_batch)
				else:
					[o1, o2] = self.convnet_.train_fn(X_batch, Y_batch)
				
				loss += o1
				acc += o2

				nbatch += 1

				if nbatch > total_batches:
					break;

			loss /= nbatch
			acc /= nbatch
			etime = time.time() - start_time

			self.elapsed_times.append(etime)
			self.losses.append(loss)
			self.accs.append(acc)


			# Evaluate
			[test_loss, test_acc] = self.convnet_.evaluate(X_test, Y_test, batch_size=512)
			self.test_losses.append(test_loss)
			self.test_accs.append(test_acc)

			[tgt_loss, tgt_acc] = self.convnet_.evaluate(X_tgt, Y_tgt, batch_size=512)
			self.tgt_losses.append(tgt_loss)
			self.tgt_accs.append(tgt_acc)

			
			print('Epoch %d: AE (Loss %.4f), Training (Loss: %.4f, Acc: %.4f), Test (Loss: %.4f, Acc: %.4f), Target (Loss: %.4f, Acc: %.4f), Time: %.2f seconds'
				% ((epoch+1), loss_ae, loss, acc, test_loss, test_acc, tgt_loss, tgt_acc, etime))


			if PREDICTPREFIX is not None:
				show_filter(X_tgt[:100], grayscale=True, filename=PREDICTPREFIX+'X-tgt-orig.png')
				show_filter(self.convae_.pred_fn(X_tgt[:100]), grayscale=True, filename=PREDICTPREFIX+'X-tgt-pred.png')
				show_filter(X[:100], grayscale=True, filename=PREDICTPREFIX+'X-src-orig.png')
				show_filter(self.convae_.pred_fn(X[:100]), grayscale=True, filename=PREDICTPREFIX+'X-src-pred.png')



			self.res = {
				'losses_ae': self.losses_ae,
				'losses':self.losses,
				'accs': self.accs,
				'test_losses': self.test_losses,
				'test_accs': self.test_accs,
				'tgt_losses': self.tgt_losses,
				'tgt_accs': self.tgt_accs,
				'elapsed_times':self.elapsed_times,
				'net_config':self.net_config,
				'ae_config':self.ae_config
			}

			if RESFILE is not None:
				pickle.dump(self.res, gzip.open(RESFILE,'wb'))

			if epoch % 10 == 0:
				print('=== > Save weights !')
				self.save_weights(PARAMFILE)
		# end epoch




	def save_weights(self, PARAMFILE):
		params_net = lasagne.layers.get_all_param_values(self.convnet_.model_)
		params_ae = lasagne.layers.get_all_param_values(self.convae_.model_)
		pickle.dump( (params_net, params_ae), gzip.open(PARAMFILE, 'wb'))


	def load_weights(self, PARAMFILE):
		(params_net, params_ae) = pickle.load(gzip.open(PARAMFILE, 'rb'))
		lasagne.layers.set_all_param_values(self.convnet_.model_, params_net)
		lasagne.layers.set_all_param_values(self.convae_.model_, params_ae)




