import gzip

import cPickle as pickle
import numpy as np

def load_mnist32x32(dataset='/local/scratch/gif/dataset/MNIST/mnist32x32.pkl.gz'):
    """
    Load MNIST handwritten digit images in 32x32.
    
    Return:
	(train_input, train_output), (validation_input, validation_output), (test_input, test_output)
	
	in [n, d1, d2, c] format
    """

    # Load images, in [n, c, d1, d2]
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
  
    train_set_y = train_set_y.astype('uint8')
    test_set_y = test_set_y.astype('uint8')
    valid_set_y = valid_set_y.astype('uint8')
    
    # Reshape to [n, d1, d2, c]
    c = 1
    d1 = 32
    d2 = 32
    ntrain = train_set_x.shape[0]
    nvalid = valid_set_x.shape[0]
    ntest = test_set_x.shape[0]

    train_set_x = np.reshape(train_set_x, (ntrain, d1, d2, c)).astype('float32')
    valid_set_x = np.reshape(valid_set_x, (nvalid, d1, d2, c)).astype('float32')
    test_set_x = np.reshape(test_set_x, (ntest, d1, d2, c)).astype('float32')
    
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)

def load_svhn(dataset='/local/scratch/gif/dataset/SVHN/svhn_gray.pkl.gz'):
    """
    Load grayscaled SVHN digit images 

    Return:
	(train_input, train_output), (test_input, test_output)
    	
	in [n, d1, d2, c] format
    """
    f = gzip.open(dataset,'rb')
    (X_train, y_train), (X_test, y_test) = pickle.load(f)
    f.close()

    idx10 = np.where(y_train == 10)
    y_train[idx10] = 0

    idx10 = np.where(y_test == 10)
    y_test[idx10] = 0
   
    X_train = X_train.astype('float32')
    y_train = y_train.astype('uint8') 
    X_test = X_test.astype('float32')
    y_test = y_test.astype('uint8')

    # Reshaep to [n, d1, d2, c]
    [ntrain, c, d1, d2] = X_train.shape
    ntest = X_test.shape[0]

    X_train = np.reshape(X_train, (ntrain, d1, d2, c))
    X_test = np.reshape(X_test, (ntest, d1, d2, c))
	
    return (X_train, y_train), (X_test, y_test)
