import gzip

import cPickle as pickle
import numpy as np

def load_mnist32x32():
    # dataset='I:\Data\PhD Life\Tutorial\Python\data\mnist.pkl.gz' # linux
    # dataset='/u/students/gif/Desktop/PhD/Tutorial/dataset/MNIST/mnist32x32.pkl.gz' # linux
    dataset = '/local/scratch/gif/dataset/MNIST/mnist32x32.pkl.gz' #the-villa
    # dataset = 'I:\Data\PhD Life\Tutorial\dataset\MNIST\mnist32x32.pkl.gz' # laptop
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

def load_usps():
    # dataset = 'I:\Data\PhD Life\Tutorial\dataset\USPS\usps.pkl.gz' #windows
    dataset='/u/students/gif/Desktop/PhD/Tutorial/dataset/USPS/usps.pkl.gz' # linux
    f = gzip.open(dataset,'rb')
    train_set, test_set = pickle.load(f)
    f.close()
    
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    


    train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, 28, 28).astype('float32')
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, 28, 28).astype('float32')
        
    rval = (train_set_x, train_set_y), (test_set_x, test_set_y)
    return rval

def load_svhn():
    # dataset = '/u/students/gif/Desktop/PhD/Tutorial/dataset/SVHN/svhn_gray.pkl.gz' #linux
    dataset = '/local/scratch/gif/dataset/SVHN/svhn_gray.pkl.gz' #the-villa
    # dataset = 'I:\Data\PhD Life\Tutorial\dataset\SVHN\svhn_gray.pkl.gz' # laptop
    f = gzip.open(dataset,'rb')
    (X_train, y_train), (X_test, y_test) = pickle.load(f)
    f.close()

    idx10 = np.where(y_train == 10)
    y_train[idx10] = 0

    idx10 = np.where(y_test == 10)
    y_test[idx10] = 0
    
    return (X_train.astype('float32'), y_train.astype('uint8')), (X_test.astype('float32'), y_test.astype('uint8'))