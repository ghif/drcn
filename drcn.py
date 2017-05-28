"""
DRCN main class 

Dependency: keras lib

Author: Muhammad Ghifary (mghifary@gmail.com)
"""

from keras.models import Model
from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import time

from myutils import * # contains all helpers for DRCN


class DRCN(object):
	def __init__(self, name='svhn-mnist'):
		"""
		Class constructor
		"""
		self.name = name

	def create_convnet(self, _input, dense_dim=1000, dy=10, nb_filters=[64, 128], kernel_size=(3, 3), pool_size=(2, 2), 
		dropout=0.5, bn=True, output_activation='softmax', opt='adam'):

		"""
		Create convnet model / encoder of DRCN

		Args:
			_input (Tensor)    	   : input layer
			dense_dim (int)            : dimensionality of the final dense layers 
			dy (int)	   	   : output dimensionality
			nb_filter (list)   	   : list of #Conv2D filters
			kernel_size (tuple)	   : Conv2D kernel size
			pool_size (tuple)  	   : MaxPool kernel size
			dropout (float)    	   : dropout rate
			bn (boolean)	   	   : batch normalization mode
			output_activation (string) : act. function for output layer
			opt (string)		   : optimizer
		
		Store the shared layers into self.enc_functions list
		"""
				
		_h = _input

		self.enc_functions = [] # to store the shared layers, will be used later for constructing conv. autoencoder
		for i, nf in enumerate(nb_filters):
			enc_f = Conv2D(nf, kernel_size, padding='same')
			_h = enc_f(_h)
			self.enc_functions.append(enc_f)
			
			_h = Activation('relu')(_h)
	
			if i < 2:
				_h = MaxPooling2D(pool_size=pool_size, padding='same')(_h)

		_h = Flatten()(_h)		
		
		enc_f = Dense(dense_dim)
		_h = enc_f(_h) 
		self.enc_functions.append(enc_f)
		if bn:
			_h = BatchNormalization()(_h)
		_h = Activation('relu')(_h)
		_h = Dropout(dropout)(_h)

		enc_f = Dense(dense_dim)
		_h = enc_f(_h)
		self.enc_functions.append(enc_f)
		if bn:
			_h = BatchNormalization()(_h)
		_feat = Activation('relu')(_h)
		_h = Dropout(dropout)(_feat)

		_y = Dense(dy, activation=output_activation)(_h)

		# convnet
		self.convnet_model = Model(input=_input, output=_y)
		self.convnet_model.compile(loss='categorical_crossentropy', optimizer=opt)
		print(self.convnet_model.summary())
		
		self.feat_model = Model(input=_input, output=_feat)
		
		
	def create_model(self, input_shape=(1, 32, 32), dense_dim=1000, dy=10, nb_filters=[64, 128], kernel_size=(3, 3), pool_size=(2, 2), 
		dropout=0.5, bn=True, output_activation='softmax', opt='adam'):
		"""
		Create DRCN model: convnet model followed by conv. autoencoder

		Args:
			_input (Tensor)    	   : input layer
			dense_dim (int)            : dimensionality of the final dense layers 
			dy (int)	   	   : output dimensionality
			nb_filter (list)   	   : list of #Conv2D filters
			kernel_size (tuple)	   : Conv2D kernel size
			pool_size (tuple)  	   : MaxPool kernel size
			dropout (float)    	   : dropout rate
			bn (boolean)	   	   : batch normalization mode
			output_activation (string) : act. function for output layer
			opt (string)		   : optimizer
		"""	
		[d1, d2, c] = input_shape

		if opt == 'adam':
			opt = Adam(lr=3e-4)
		elif opt == 'rmsprop':
			opt = RMSprop(lr=1e-4)


		_input = Input(shape=input_shape)
		
		# Create ConvNet
		self.create_convnet(_input, dense_dim=dense_dim, dy=dy, nb_filters=nb_filters, 
			kernel_size=kernel_size, pool_size=pool_size, dropout=dropout, 
			bn=bn, output_activation=output_activation, opt=opt) 
		
		# Create ConvAE, encoder functions are shared with ConvNet
		_h = _input
		
		# Reconstruct Conv2D layers
		for i, nf in enumerate(nb_filters):
			_h = self.enc_functions[i](_h)
			_h = Activation('relu')(_h)
			if i < 2:
				_h = MaxPooling2D(pool_size=pool_size, padding='same')(_h)


		[_, wflat, hflat, cflat] = _h.get_shape().as_list()	
		_h = Flatten()(_h)
		
		# Dense layers
		for i in range(len(nb_filters), len(self.enc_functions)):
			_h = self.enc_functions[i](_h)
			_h = Activation('relu')(_h)
			
		# Decoder
		_h = Dense(dense_dim)(_h)
		_h = Activation('relu')(_h)
		
		_xdec = Dense(wflat*hflat*cflat)(_h)
		_xdec = Activation('relu')(_xdec)
		_xdec = Reshape((wflat, hflat, nb_filters[-1]))(_xdec)
		i = 0
		for nf in reversed(nb_filters):
			_xdec = Conv2D(nf, kernel_size, padding='same')(_xdec)
			_xdec = Activation('relu')(_xdec)
			
			if i > 0:
				_xdec = UpSampling2D(size=pool_size)(_xdec)	
			i += 1
		
		_xdec = Conv2D(c, kernel_size, padding='same', activation=clip_relu)(_xdec)	

		self.convae_model = Model(input=_input, output=_xdec)	
		self.convae_model.compile(loss='mse', optimizer=opt)
		print(self.convae_model.summary())
	
	def fit_drcn(self, X, Y, Xu, nb_epoch=50, batch_size=128, shuffle=True,
			validation_data=None, test_data=None, PARAMDIR=None, CONF=None):
		"""
		DRCN algorithm: 	
			- i) train convnet on labeled source data, ii) train convae on unlabeled target data
			- include data augmentation and denoising
		
		Args:
			X (np.array)  	  	: [n, d1, d2, c] array of source images
			Y (np.array)  	  	: [n, dy] array of source labels
			Xu (np.array) 	  	: [n, d1, d2, c] array of target images
			nb_epoch (int)	  	: #iteration of gradient descent
			batch_size (int)  	: # data per batch
			shuffle (boolean) 	: shuffle the data in a batch if True
			validation_data (tuple) : tuple of (Xval, Yval) array
			test_data	  	: tuple of (Xtest, Ytest) array
			PARAMDIR (string)	: directory to store the learned weights
			CONF (string)		: for naming purposes
			
			
		"""
		history = {}
		history['losses'] = []
		history['accs'] = []
		history['gen_losses'] = []
		history['val_losses'] = []
		history['val_accs'] = []
		history['test_losses'] = []
		history['test_accs'] = []
		history['elapsed_times'] = []

		best_ep = 1
		
		# data augmenter and batch iterator for each convnet and convae
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
			    rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
			    width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
			    height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
			    horizontal_flip=False, # randomly flip images
			    vertical_flip=False
			) # randomly flip images


		for e in range(nb_epoch):
			start_t = time.time()
			# convae training
			gen_loss = 0.
			n_batch = 0
			total_batches = Xu.shape[0] / batch_size

			for Xu_batch, Yu_batch in gdatagen.flow(Xu, np.copy(Xu), batch_size=batch_size, shuffle=shuffle):

				Xu_batch = get_impulse_noise(Xu_batch, 0.5)

				l = self.convae_model.train_on_batch(Xu_batch, Yu_batch)
				gen_loss += l
				n_batch += 1
				
				if n_batch >= total_batches:
					break
			
			gen_loss /= n_batch
			history['gen_losses'].append(gen_loss)

			# convnet training
			loss = 0.
			n_batch = 0
			total_batches = X.shape[0] / batch_size 

			for X_batch, Y_batch in ddatagen.flow(X, Y, batch_size=batch_size, shuffle=shuffle):				
				l = self.convnet_model.train_on_batch(X_batch, Y_batch)
				loss += l
				n_batch += 1
				
				if n_batch >= total_batches:
					break

				

			loss /= n_batch
			history['losses'].append(loss)

			# calculate accuracy
			acc = accuracy(self.convnet_model.predict(X), Y)
			history['accs'].append(acc)

						
			elapsed_t = time.time() - start_t
			history['elapsed_times'].append(elapsed_t)

						
			val_loss = -1
			val_acc = -1
			best_val_acc = -1
			if validation_data is not None:
				(X_val, Y_val) = validation_data
				val_loss = 0.
				n_batch = 0
				for Xv, Yv in iterate_minibatches(X_val, Y_val ,batch_size, shuffle=False):
					l = self.convnet_model.test_on_batch(Xv, Yv)
					val_loss += l
					n_batch += 1
				val_loss /= n_batch
				history['val_losses'].append(val_loss)
				
				val_acc = accuracy(self.convnet_model.predict(X_val), Y_val)
				history['val_accs'].append(val_acc)
			
			test_loss = -1
			test_acc = -1
			if test_data is not None:
				(X_test, Y_test) = test_data
				test_loss = 0.
				n_batch = 0
				for Xt, Yt in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
					l = self.convnet_model.test_on_batch(Xt, Yt)
					test_loss += l
					n_batch += 1

				test_loss /= n_batch
				history['test_losses'].append(test_loss)
				
				test_acc = accuracy(self.convnet_model.predict(X_test), Y_test)
				history['test_accs'].append(test_acc)

									
					 
			print('Epoch-%d: (loss: %.3f, acc: %.3f, gen_loss: %.3f), (val_loss: %.3f, val_acc: %.3f), (test_Loss: %.3f, test_acc: %.3f) -- %.2f sec' % \
				((e+1), loss, acc, gen_loss, val_loss, val_acc, test_loss, test_acc,  elapsed_t))

			if PARAMDIR is not None:
				if (acc + val_acc) > best_val_acc:
					best_val_acc = (acc + val_acc)
					best_ep = e + 1
					CONFCNN ='%s_cnn' % CONF
					save_weights(self.convnet_model, PARAMDIR, CONFCNN)

					CONFCAE = '%s_cae' % CONF
					save_weights(self.convae_model, PARAMDIR, CONFCAE)
				else:
					print('do not save, best val_acc: %.3f at %d' % (best_val_acc, best_ep))


			# store history
			HISTPATH = '%s_hist.npy' % CONF
			np.save(HISTPATH, history)

			# visualization 
			if validation_data is not  None:
				(X_val, Y_val) = validation_data
				Xsv = X_val[:100]

				Xs = postprocess_images(Xsv, omin=0, omax=1)
				imgfile = '%s_src.png' % CONF
				Xs = np.reshape(Xs, (len(Xs), Xs.shape[3], Xs.shape[1], Xs.shape[2]))
				show_images(Xs, filename=imgfile)
				
				Xs_pred = self.convae_model.predict(Xsv)
				Xs_pred = postprocess_images(Xs_pred, omin=0, omax=1)
				imgfile = '%s_src_pred.png' % CONF
				Xs_pred = np.reshape(Xs_pred, (len(Xs_pred), Xs_pred.shape[3], Xs_pred.shape[1], Xs_pred.shape[2]))
				show_images(Xs_pred, filename=imgfile)

			if test_data is not  None:
				(X_test, Y_test) = test_data
				Xtv = X_test[:100]
				Xt = postprocess_images(Xtv, omin=0, omax=1)
				imgfile = '%s_tgt.png' % CONF
				Xt = np.reshape(Xt, (len(Xt), Xt.shape[3], Xt.shape[1], Xt.shape[2]))
				show_images(Xt, filename=imgfile)
				
				Xt_pred = self.convae_model.predict(Xtv)
				Xt_pred = postprocess_images(Xt_pred, omin=0, omax=1)
				imgfile = '%s_tgt_pred.png' % CONF
				Xt_pred = np.reshape(Xt_pred, (len(Xt_pred), Xt_pred.shape[3], Xt_pred.shape[1], Xt_pred.shape[2]))
				show_images(Xt_pred, filename=imgfile)


	###  just in case want to run convnet and convae separately, below are the training modules  ###
	def fit_convnet(self, X, Y, nb_epoch=50, batch_size=128, shuffle=True,
			validation_data=None, test_data=None, PARAMDIR=None, CONF=None):
			
		history = {}
		history['losses'] = []
		history['accs'] = []
		history['val_losses'] = []
		history['val_accs'] = []
		history['test_losses'] = []
		history['test_accs'] = []
		history['elapsed_times'] = []

		best_ep = 1
		for e in range(nb_epoch):
			loss = 0.
			n_batch = 0
			start_t = time.time()
			for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=shuffle):

				l = self.convnet_model.train_on_batch(X_batch, Y_batch)
				loss += l
				n_batch += 1
			
			elapsed_t = time.time() - start_t
			history['elapsed_times'].append(elapsed_t)

			loss /= n_batch
			history['losses'].append(loss)

			# calculate accuracy
			acc = accuracy(self.convnet_model.predict(X), Y)
			history['accs'].append(acc)
			
			
			val_loss = -1
			val_acc = -1
			best_val_acc = -1
			if validation_data is not None:
				(X_val, Y_val) = validation_data
				val_loss = 0.
				n_batch = 0
				for Xv, Yv in iterate_minibatches(X_val, Y_val ,batch_size, shuffle=False):
					l = self.convnet_model.test_on_batch(Xv, Yv)
					val_loss += l
					n_batch += 1
				val_loss /= n_batch
				history['val_losses'].append(val_loss)
				
				val_acc = accuracy(self.convnet_model.predict(X_val), Y_val)
				history['val_accs'].append(val_acc)
			
			test_loss = -1
			test_acc = -1
			if test_data is not None:
				(X_test, Y_test) = test_data
				test_loss = 0.
				n_batch = 0
				for Xt, Yt in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
					l = self.convnet_model.test_on_batch(Xt, Yt)
					test_loss += l
					n_batch += 1

				test_loss /= n_batch
				history['test_losses'].append(test_loss)
				
				test_acc = accuracy(self.convnet_model.predict(X_test), Y_test)
				history['test_accs'].append(test_acc)

									
					 
			print('Epoch-%d: (loss: %.3f, acc: %.3f), (val_loss: %.3f, val_acc: %.3f), (test_Loss: %.3f, test_acc: %.3f) -- %.2f sec' % \
				((e+1), loss, acc, val_loss, val_acc, test_loss, test_acc,  elapsed_t))

			if PARAMDIR is not None:
				if (acc + val_acc) > best_val_acc:
					best_val_acc = (acc + val_acc)
					best_ep = e + 1
					save_weights(self.convnet_model, PARAMDIR, CONF)
				else:
					print('do not save, best val_acc: %.3f at %d' % (best_val_acc, best_ep))


			# store history
			HISTPATH = '%s_hist.npy' % CONF
			np.save(HISTPATH, history)

		
	
	def fit_convae(self, X, nb_epoch=50, batch_size=128, shuffle=True,
		validation_data=None,  test_data=None, PARAMDIR=None, CONF=None):
		
		
		history = {}
		history['losses'] = []
		history['val_losses'] = []		
		history['test_losses'] = []
		history['elapsed_times'] = []

		best_ep = 1
		for e in range(nb_epoch):
			loss = 0.
			n_batch = 0
			start_t = time.time()
			for X_batch, Y_batch in iterate_minibatches(X, np.copy(X), batch_size, shuffle=shuffle):

				l = self.convae_model.train_on_batch(X_batch, Y_batch)
				loss += l
				n_batch += 1
			
			elapsed_t = time.time() - start_t
			history['elapsed_times'].append(elapsed_t)

			loss /= n_batch
			history['losses'].append(loss)

						
			
			val_loss = -1
			best_val_loss = 100000
	
			test_loss = -1	
									
					 
			print('Epoch-%d: (loss: %.3f), (val_loss: %.3f), (test_Loss: %.3f) -- %.2f sec' % \
				((e+1), loss, val_loss,test_loss,elapsed_t))

			if PARAMDIR is not None:
				if loss < best_val_loss:
					best_val_loss = loss
					best_ep = e + 1
					save_weights(self.convae_model, PARAMDIR, CONF)
				else:
					print('do not save, best val loss: %.3f at %d' % (best_val_loss, best_ep))


			# store history
			HISTPATH = '%s_hist.npy' % CONF
			np.save(HISTPATH, history)

			# visualization 
			if validation_data is not  None:
				Xtv = validation_data
				Xt = postprocess_images(Xtv, omin=0, omax=1)
				imgfile = '%s_tgt.png' % CONF
				Xt = np.reshape(Xt, (len(Xt), Xt.shape[3], Xt.shape[1], Xt.shape[2]))

				show_images(Xt, filename=imgfile)
				
				Xt_pred = self.convae_model.predict(Xtv)
				Xt_pred = postprocess_images(Xt_pred, omin=0, omax=1)
				imgfile = '%s_tgt_pred.png' % CONF

				Xt_pred = np.reshape(Xt_pred, (len(Xt_pred), Xt_pred.shape[3], Xt_pred.shape[1], Xt_pred.shape[2]))
				show_images(Xt_pred, filename=imgfile)

			if test_data is not  None:
				Xsv = test_data
				Xs = postprocess_images(Xsv, omin=0, omax=1)
				imgfile = '%s_src.png' % CONF
				Xs = np.reshape(Xs, (len(Xs), Xs.shape[3], Xs.shape[1], Xs.shape[2]))
				show_images(Xs, filename=imgfile)
				
				Xs_pred = self.convae_model.predict(Xsv)
				Xs_pred = postprocess_images(Xs_pred, omin=0, omax=1)
				imgfile = '%s_src_pred.png' % CONF
				Xs_pred = np.reshape(Xs_pred, (len(Xs_pred), Xs_pred.shape[3], Xs_pred.shape[1], Xs_pred.shape[2]))
				show_images(Xs_pred, filename=imgfile)

	
