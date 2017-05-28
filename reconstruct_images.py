"""
Display DRCN's reconstruction results
"""
import numpy as np
import gzip
import cPickle as pickle

from keras.utils import np_utils

from drcn import *
from myutils import *

from dataset import *

def getImagesPerClass(X, y, nd=10, nc=10):
	"""
	Take only 'nd' images from each class
	"""
	print('y :', y)
	xlist = []
	ylist = []
	for c in range(nc):
		idx, = np.where(y == c)
		idx = idx[:10]
		xlist.append(X[idx])
		ylist.append(y[idx])


	Xs = np.concatenate(xlist, axis=0)
	Ys = np.concatenate(ylist, axis=0)
	return (Xs, Ys)
	
		
	

# Load datasets
print('Load datasets')
(X_train, y_train), (X_test, y_test) = load_svhn(dataset='/local/scratch/gif/dataset/SVHN/svhn_gray.pkl.gz') # source

nb_classes = 10
(Xs, _) = getImagesPerClass(X_train, y_train, nd=10, nc=nb_classes)
Xsv = preprocess_images(Xs, tmin=0, tmax=1)


print('Load trained DRCN')
drcn = DRCN()
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
drcn.create_model(input_shape=input_shape, dense_dim=1024, dy=nb_classes, nb_filters=[100, 150, 200], kernel_size=(3, 3), pool_size=(2, 2), 
		dropout=0.5, bn=False, output_activation='softmax', opt='adam')
PARAMDIR = ''
CONF = 'svhn-mnist_drcn_v2_cae'
PARAMPATH = os.path.join(PARAMDIR, '%s_weights.h5') % CONF
drcn.convae_model.load_weights(PARAMPATH)

print('Store images ...')
Xs = postprocess_images(Xsv, omin=0, omax=1)
imgfile = '%s_src.png' % CONF
Xs = np.reshape(Xs, (len(Xs), Xs.shape[3], Xs.shape[1], Xs.shape[2]))
show_images(Xs, filename=imgfile)
				
Xs_pred = drcn.convae_model.predict(Xsv)
Xs_pred = postprocess_images(Xs_pred, omin=0, omax=1)
imgfile = '%s_src_pred.png' % CONF
Xs_pred = np.reshape(Xs_pred, (len(Xs_pred), Xs_pred.shape[3], Xs_pred.shape[1], Xs_pred.shape[2]))
show_images(Xs_pred, filename=imgfile)

