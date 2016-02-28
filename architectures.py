from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import activity_l1



# def get_svhn_mnist_convnet():
# 	model = Sequential()
# 	## ARCHITECTURE ##
# 	# Masci et al. 2011
# 	model.add(Convolution2D(100, 1, 5, 5, border_mode='full')) #0
# 	model.add(Activation('relu')) #1
# 	model.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	model.add(Convolution2D(150, 100, 5, 5)) #3
# 	model.add(Activation('relu')) #4
# 	model.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	model.add(Convolution2D(200, 150, 3, 3)) #6
# 	model.add(Activation('relu')) #7

# 	model.add(Flatten()) #11
# 	model.add(Dense(200*5*5, 1024)) #12
# 	model.add(Activation('relu')) #13
# 	model.add(Dropout(0.5)) #14
# 	# model.add(Dropout(0.1)) #10

# 	model.add(Dense(1024, 1024)) #12
# 	model.add(Activation('relu')) #13
# 	model.add(Dropout(0.5)) #14


# 	model.add(Dense(1024, 10)) #15
# 	model.add(Activation('softmax')) #12

# 	return model

# def get_svhn_mnist_convae():
# 	gmodel = Sequential() 
# 	gmodel.add(Convolution2D(100, 1, 5, 5, border_mode='full') )  #0
# 	gmodel.add(Activation('relu')) #1
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #2
# 	# gmodel.add(Dropout(0.25)) 

# 	gmodel.add(Convolution2D(150, 100, 5, 5)) #3
# 	gmodel.add(Activation('relu')) #4
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #5
# 	# gmodel.add(Dropout(0.5)) 

# 	gmodel.add(Convolution2D(200, 150, 3, 3)) #6
# 	gmodel.add(Activation('relu')) #7
# 	# gmodel.add(Dropout(0.5)) 

# 	gmodel.add(Flatten()) #8
# 	gmodel.add(Dense(200*5*5, 1024)) #9
# 	gmodel.add(Activation('relu')) #10
# 	# gmodel.add(Dropout(0.5)) 

# 	gmodel.add(Dense(1024, 1024)) #9
# 	gmodel.add(Activation('relu')) #10

# 	gmodel.add(Dense(1024,200*5*5)) #11
# 	gmodel.add(Activation('relu')) #12
# 	gmodel.add(Reshape(200,5,5)) #13
# 	# gmodel.add(Dropout(0.5)) 

# 	gmodel.add(Convolution2D(150, 200, 3, 3, border_mode='full'))  #14
# 	gmodel.add(Activation('relu')) #15
# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #16
# 	# gmodel.add(Dropout(0.5)) 

# 	gmodel.add(Convolution2D(100, 150, 5, 5, border_mode='full')) #17
# 	gmodel.add(Activation('relu')) #18
# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #19
# 	# gmodel.add(Dropout(0.25)) 

# 	gmodel.add(Convolution2D(1, 100, 5, 5)) #20
# 	gmodel.add(Activation('linear')) #21

# 	return gmodel


def get_convnet(dropout_rate=0.3):
	model = Sequential()
	## ARCHITECTURE ##
	# Masci et al. 2011
	model.add(Convolution2D(100, 5, 5, border_mode='same', input_shape=(1, 32, 32), init='glorot_uniform' )) #0
	model.add(Activation('relu')) #1
	model.add(MaxPooling2D(pool_size=(2, 2))) #2

	model.add(Convolution2D(150, 5, 5, init='glorot_uniform')) #3
	model.add(Activation('relu')) #4
	model.add(MaxPooling2D(pool_size=(2, 2))) #5

	model.add(Convolution2D(200, 3, 3, init='glorot_uniform')) #6
	model.add(Activation('relu')) #7

	model.add(Flatten()) #11
	model.add(Dense(300, init='glorot_uniform')) #12
	model.add(Activation('relu')) #13
	if dropout_rate > 0.0:
		model.add(Dropout(dropout_rate)) #14
	
	model.add(Dense(10, init='glorot_uniform')) #15
	model.add(Activation('softmax')) #12

	return model

# def get_mnist_ssl_convnet():
# 	## DISCRIMINATIVE MODEL ##
# 	# LeNet
# 	dmodel = Sequential() 
# 	dmodel.add(Convolution2D(32, 1, 5, 5, border_mode='full'))  #0
# 	dmodel.add(Activation('relu')) #1
# 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	dmodel.add(Convolution2D(32, 32, 5, 5)) #3
# 	dmodel.add(Activation('relu')) #4
# 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	dmodel.add(Flatten()) #6
# 	dmodel.add(Dropout(0.5)) #7
# 	dmodel.add(Dense(32*6*6, 256)) #8
# 	dmodel.add(Activation('relu')) #9

# 	dmodel.add(Dropout(0.5)) #10
# 	dmodel.add(Dense(256, 10)) #11
# 	dmodel.add(Activation('softmax')) #12

# 	##############

# 	return dmodel

# def get_mnist_ssl_convae():
# 	## DISCRIMINATIVE MODEL ##
# 	# LeNet
# 	gmodel = Sequential() 
# 	gmodel.add(Convolution2D(32, 1, 5, 5, border_mode='full'))  #0
# 	gmodel.add(Activation('relu')) #1
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	gmodel.add(Convolution2D(32, 32, 5, 5)) #3
# 	gmodel.add(Activation('relu')) #4
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	gmodel.add(Flatten()) #6
# 	gmodel.add(Dense(32*6*6, 256)) #7
# 	gmodel.add(Activation('relu')) #8

# 	gmodel.add(Dense(256, 256)) #9
# 	gmodel.add(Activation('relu')) #10

# 	gmodel.add(Dense(256,32*6*6)) #11
# 	gmodel.add(Activation('relu')) #12
# 	gmodel.add(Reshape(32,6,6)) #13
	
# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #14
# 	gmodel.add(Convolution2D(32, 32, 5, 5, border_mode='full'))  #15
# 	gmodel.add(Activation('relu')) #16
	
# 	# gmodel.add(Dropout(0.5)) 
# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #17
# 	gmodel.add(Convolution2D(1, 32, 5, 5)) #18
# 	gmodel.add(Activation('linear')) #19
# 	##############

# 	return gmodel

# def get_mnist_convae_shallow():
# 	gmodel = Sequential() 
# 	gmodel.add(Convolution2D(32, 1, 5, 5, border_mode='full')) 
# 	gmodel.add(Activation('relu')) 
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #
	
# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #unpooling
# 	gmodel.add(Convolution2D(1, 32, 5, 5)) #deconv
	
# 	gmodel.add(Activation('relu'))
# 	##############

# 	return gmodel

# # def get_mnist_usps_dmdgn():
# # 	## DISCRIMINATIVE MODEL ##
# # 	# Masci et al. 2011
# # 	dmodel = Sequential() 
# # 	dmodel.add(Convolution2D(100, 1, 5, 5, border_mode='full'))  #0
# # 	dmodel.add(Activation('relu')) #1
# # 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# # 	dmodel.add(Convolution2D(150, 100, 5, 5)) #3
# # 	dmodel.add(Activation('relu')) #4
# # 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# # 	dmodel.add(Convolution2D(200, 150, 3, 3)) #6
# # 	dmodel.add(Activation('relu')) #7

# # 	dmodel.add(Flatten()) #8
# # 	dmodel.add(Dense(200*4*4, 300)) #9
# # 	dmodel.add(Activation('relu')) #10
# # 	dmodel.add(Dropout(0.5)) #11

# # 	dmodel.add(Dense(300, 10)) #12
# # 	dmodel.add(Activation('softmax')) #13
# # 	##############

# # 	# opt = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-6)
# # 	opt = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-6)
# # 	dmodel.compile(loss='categorical_crossentropy', optimizer=opt)
# # 	##############

# # 	return dmodel


# def get_mnist_usps_convnet():
# 	## DISCRIMINATIVE MODEL ##
# 	# Masci et al. 2011
# 	dmodel = Sequential() 
# 	dmodel.add(Convolution2D(100, 1, 5, 5, border_mode='full'))  #0
# 	dmodel.add(Activation('relu')) #1
# 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	dmodel.add(Convolution2D(150, 100, 5, 5)) #3
# 	dmodel.add(Activation('relu')) #4
# 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	dmodel.add(Convolution2D(200, 150, 3, 3)) #6
# 	dmodel.add(Activation('relu')) #7
# 	dmodel.add(Flatten()) #8
# 	dmodel.add(Dropout(0.5))

# 	dmodel.add(Dense(200*4*4, 300)) #9
# 	dmodel.add(Activation('relu')) #10
# 	dmodel.add(Dropout(0.5)) #11

# 	dmodel.add(Dense(300, 10)) #12
# 	dmodel.add(Activation('softmax')) #13

# 	##############

# 	return dmodel

# def get_mnist_usps_convae():
# 	## GENERATIVE MODEL ##
# 	# Masci et al. 2011
# 	gmodel = Sequential() 
# 	gmodel.add(Convolution2D(100, 1, 5, 5, border_mode='full') )  #0
# 	gmodel.add(Activation('relu')) #1
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	gmodel.add(Convolution2D(150, 100, 5, 5)) #3
# 	gmodel.add(Activation('relu')) #4
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	gmodel.add(Convolution2D(200, 150, 3, 3)) #6
# 	gmodel.add(Activation('relu')) #7
# 	gmodel.add(Flatten()) #8
# 	# gmodel.add(Dropout(0.5))

# 	gmodel.add(Dense(200*4*4, 300)) #9
# 	gmodel.add(Activation('relu')) #10
# 	# gmodel.add(Dropout(0.5))
	
# 	gmodel.add(Dense(300,200*4*4)) #11
# 	gmodel.add(Activation('relu')) #12
# 	gmodel.add(Reshape(200,4,4)) #13

# 	gmodel.add(Convolution2D(150, 200, 3, 3, border_mode='full'))  #14
# 	gmodel.add(Activation('relu')) #15

# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #16
# 	gmodel.add(Convolution2D(100, 150, 5, 5, border_mode='full')) #17
# 	gmodel.add(Activation('relu')) #18

# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #19
# 	gmodel.add(Convolution2D(1, 100, 5, 5)) #20
# 	gmodel.add(Activation('hard_sigmoid')) #21
# 	##############

# 	return gmodel


# def get_stl_cifar_convnet(nb_classes=8):
# 	## DISCRIMINATIVE MODEL ##
# 	# Masci et al. 2011
# 	dmodel = Sequential() 
# 	dmodel.add(Convolution2D(100, 3, 5, 5, border_mode='full'))  #0
# 	dmodel.add(Activation('relu')) #1
# 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	dmodel.add(Convolution2D(150, 100, 5, 5)) #3
# 	dmodel.add(Activation('relu')) #4
# 	dmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	dmodel.add(Convolution2D(200, 150, 3, 3)) #6
# 	dmodel.add(Activation('relu')) #7

# 	dmodel.add(Flatten()) #8
	
# 	dmodel.add(Dropout(0.5)) #11
# 	dmodel.add(Dense(200*5*5, 300)) #9
# 	dmodel.add(Activation('relu')) #10
	
# 	dmodel.add(Dropout(0.5)) #11
# 	dmodel.add(Dense(300, nb_classes)) #12
# 	dmodel.add(Activation('softmax')) #13

# 	##############

# 	return dmodel


# def get_stl_cifar_convae():
# 	## GENERATIVE MODEL ##
# 	# Masci et al. 2011
# 	gmodel = Sequential() 
# 	gmodel.add(Convolution2D(100, 3, 5, 5, border_mode='full') )  #0
# 	gmodel.add(Activation('relu')) #1
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	gmodel.add(Convolution2D(150, 100, 5, 5)) #3
# 	gmodel.add(Activation('relu')) #4
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #5

# 	gmodel.add(Convolution2D(200, 150, 3, 3)) #6
# 	gmodel.add(Activation('relu')) #7
# 	gmodel.add(Flatten()) #8
# 	# gmodel.add(Dropout(0.5))

# 	gmodel.add(Dense(200*5*5, 300)) #9
# 	gmodel.add(Activation('relu')) #10
# 	# gmodel.add(Dropout(0.5))
	
# 	gmodel.add(Dense(300,200*5*5)) #11
# 	gmodel.add(Activation('relu')) #12
# 	gmodel.add(Reshape(200,5,5)) #13

# 	gmodel.add(Convolution2D(150, 200, 3, 3, border_mode='full'))  #14
# 	gmodel.add(Activation('relu')) #15

# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #16
# 	gmodel.add(Convolution2D(100, 150, 5, 5, border_mode='full')) #17
# 	gmodel.add(Activation('relu')) #18

# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #19
# 	gmodel.add(Convolution2D(3, 100, 5, 5)) #20
# 	gmodel.add(Activation('linear')) #21
# 	##############

# 	return gmodel

# def get_shallow_convae():
# 	## GENERATIVE MODEL ##
# 	# Masci et al. 2011
# 	gmodel = Sequential() 
# 	gmodel.add(Convolution2D(96, 3, 7, 7, border_mode='full', activity_regularizer=activity_l1(1e-4)) )  #0
# 	gmodel.add(Activation('relu')) #1
# 	gmodel.add(MaxPooling2D(poolsize=(2, 2))) #2

# 	gmodel.add(Unpooling2D(poolsize=(2,2))) #19
# 	gmodel.add(Convolution2D(3, 96, 7, 7)) #20
# 	gmodel.add(Activation('linear')) #21
# 	##############

# 	return gmodel