from keras.models import Model
from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, Nadam

import os
import numpy as np
import time

from myutils import *

from keras import backend as K

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
	assert len(inputs) == len(targets)

	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)

	for start_idx in range(0, len(inputs), batchsize):
		end_idx = start_idx + batchsize
		if end_idx > len(inputs):
			end_idx = start_idx + (len(inputs) % batchsize)

		if shuffle:
			excerpt = indices[start_idx:end_idx]
	
		else:
			excerpt = slice(start_idx, end_idx)

		yield inputs[excerpt], targets[excerpt]

def accuracy(Y1, Y2):
	n = Y1.shape[0]
	ntrue = np.count_nonzero(np.argmax(Y1, axis=1) == np.argmax(Y2, axis=1))
	return ntrue * 1.0 / n

def save_weights(model, PARAMDIR, CONF):
	# model: keras model
	print(' == save weights == ')

	# save weights
	PARAMPATH = os.path.join(PARAMDIR, '%s_weights.h5') % CONF
	model.save(PARAMPATH)
	
	# save architecture
	CONFPATH = os.path.join(PARAMDIR, '%s_conf.json') % CONF
	archjson = model.to_json()

	open(CONFPATH, 'wb').write(archjson)


def clip_relu(x):
	y = K.maximum(x, 0)
	return K.minimum(y, 1)

class DRCN(object):
	def __init__(self, name='svhn-mnist'):
		self.name = name

	def create_model(self, input_shape=(1, 32, 32), dy=10, nb_filters=[64, 128], kernel_size=(3, 3), pool_size=(2, 2), 
		dropout=0.5, output_activation='softmax', opt='adam'):
		
		[d1, d2, c] = input_shape

		if opt == 'adam':
			opt = Adam(lr=3e-4)
		elif opt == 'rmsprop':
			opt = RMSprop(lr=1e-4)


		_input = Input(shape=input_shape)
		
		_h = _input
		print('input_shape : ', input_shape)
		print('kernel_size: ', kernel_size)
		for i, nf in enumerate(nb_filters):
			_h = Conv2D(nf, kernel_size, padding='same')(_h)
			_h = BatchNormalization()(_h)
			_h = Activation('relu')(_h)			
			_h = Dropout(dropout)(_h)
			_h = MaxPooling2D(pool_size=pool_size, padding='same')(_h)

		# _h = AveragePooling2D(pool_size=pool_size)(_h)
		# _h = MaxPooling2D(pool_size=pool_size)(_h)
		
		_h = Flatten()(_h)
		
		_h = Dense(1000)(_h)
		_h = BatchNormalization()(_h)
		_feat = Activation('relu')(_h)
		
		_h = Dropout(dropout)(_feat)

		_y = Dense(dy, activation=output_activation)(_feat)

		# convnet
		self.convnet_model = Model(input=_input, output=_y)
		self.convnet_model.compile(loss='categorical_crossentropy', optimizer=opt)

		# shared features
		self.feat_model = Model(input=_input, output=_feat)

		print(self.convnet_model.summary())

		# conv autoencoder
		dim = 4
		_xdec = Dense(3200, activation='relu')(_feat)
		# _xdec = Reshape((nb_filters[-1], dim, dim))(_xdec)
		_xdec = Reshape((dim, dim, nb_filters[-1]))(_xdec)
		for nf in reversed(nb_filters):
			# _xdec = Conv2DTranspose(nf, kernel_size, padding='same', subsample=(2, 2))(_xdec)
			_xdec = Conv2D(nf, kernel_size, padding='same')(_xdec)
			
			# _xdec = Convolution2D(nf, kernel_size[0], kernel_size[1], border_mode='same')(_xdec)

			_xdec = BatchNormalization()(_xdec)
			_xdec = Activation('relu')(_xdec)
			# _xdec = Dropout(dropout)(_xdec)
			_xdec = UpSampling2D(size=pool_size)(_xdec)

		# _xdec = Deconvolution2D(1, kernel_size[0], kernel_size[1], (None, 1, 32, 32), subsample=(2, 2), border_mode='same', activation='tanh')(_xdec)
	 	# _xdec = Conv2DTranspose(1, kernel_size, padding='same', activation=clip_relu)(_xdec)
		_xdec = Conv2D(c, kernel_size, padding='same', activation=clip_relu)(_xdec)

		self.convae_model = Model(input=_input, output=_xdec)
				
		self.convae_model.compile(loss='mse', optimizer=opt)

		print(self.convae_model.summary())
	

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

	def fit_drcn(self, X, Y, Xu, nb_epoch=50, batch_size=128, shuffle=True,
			validation_data=None, test_data=None, PARAMDIR=None, CONF=None):
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
		for e in range(nb_epoch):
			start_t = time.time()
			# convae training
			gen_loss = 0.
			n_batch = 0
			for Xu_batch, Yu_batch in iterate_minibatches(Xu, np.copy(Xu), batch_size, shuffle=shuffle):
				l = self.convae_model.train_on_batch(Xu_batch, Yu_batch)
				gen_loss += l
				n_batch += 1
			
			gen_loss /= n_batch
			history['gen_losses'].append(gen_loss)

			# convnet training
			loss = 0.
			n_batch = 0
			
			for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=shuffle):

				l = self.convnet_model.train_on_batch(X_batch, Y_batch)
				loss += l
				n_batch += 1

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
					save_weights(self.convnet_model, PARAMDIR, CONF)
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
				show_images(Xs, filename=imgfile)
				
				Xs_pred = self.convae_model.predict(Xsv)
				Xs_pred = postprocess_images(Xs_pred, omin=0, omax=1)
				imgfile = '%s_src_pred.png' % CONF
				show_images(Xs_pred, filename=imgfile)

			if test_data is not  None:
				(X_test, Y_test) = test_data
				Xtv = X_test[:100]
				Xt = postprocess_images(Xtv, omin=0, omax=1)
				imgfile = '%s_tgt.png' % CONF
				show_images(Xt, filename=imgfile)
				
				Xt_pred = self.convae_model.predict(Xtv)
				Xt_pred = postprocess_images(Xt_pred, omin=0, omax=1)
				imgfile = '%s_tgt_pred.png' % CONF
				show_images(Xt_pred, filename=imgfile)

