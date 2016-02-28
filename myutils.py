import numpy as np
import random

from sklearn import preprocessing
from keras.utils import np_utils

def get_subsample(X, y, nc, C=10):
    # nc : number of samples per classes
    G_list = []
    L_list = []
    for c in range(0,C):
        inds_c = np.where(y == c)
        
        inds_c = inds_c[0]
        
        # inds_c = np.random.permutation(inds_c)

        G = X[inds_c]
        L = y[inds_c]

        G = G[0:nc]
        L = L[0:nc]

        G_list.append(G)
        L_list.append(L)



    X_sub = G_list[0]
    y_sub = L_list[0]
    for c in range(1,C):
        X_sub = np.concatenate((X_sub, G_list[c]), axis=0)
        y_sub = np.concatenate((y_sub, L_list[c]), axis=0)

    return X_sub, y_sub

# # model1 => model2
def copy_weights(model1, model2, l=1):
    ignored_layers = ['Activation','Dropout','MaxPooling2D','Unpooling2D']
    inds1 = []
    for l in range(0,len(model1.layers)):
        lname = model1.layers[l].get_config()["name"]
        if lname not in ignored_layers:
            # print(l,':',model1.layers[l].get_config()["name"])
            inds1.append(l)
    # print('+'*10)
    inds2 = []
    for l in range(0,len(model2.layers)):
        lname = model2.layers[l].get_config()["name"]
        if lname not in ignored_layers:
            # print(l,':',model2.layers[l].get_config()["name"])
            inds2.append(l)

    n1 = len(inds1)
    n2 = len(inds2)
    
    nmin = np.minimum(n1,n2)
    
    for i in range(0,nmin-1):
        l1 = inds1[i]
        l2 = inds2[i]
        # print(l1,l2)

        if l < 1:
            W2 = model2.layers[l2].get_weights()
            model2.layers[l2].set_weights(W2 + l*model1.layers[l1].get_weights())
        else:
            model2.layers[l2].set_weights(model1.layers[l1].get_weights())

def copy_weights_convae(model1, model2):
    ignored_layers = ['Activation','Dropout','MaxPooling2D','Unpooling2D']
    inds1 = []
    for l in range(0,len(model1.layers)):
        lname = model1.layers[l].get_config()["name"]
        if lname not in ignored_layers:
            # print(l,':',model1.layers[l].get_config()["name"])
            inds1.append(l)
    # print('+'*10)
    inds2 = []
    for l in range(0,len(model2.layers)):
        lname = model2.layers[l].get_config()["name"]
        if lname not in ignored_layers:
            # print(l,':',model2.layers[l].get_config()["name"])
            inds2.append(l)

    n1 = len(inds1)
    n2 = len(inds2)
    
    # nmin = np.minimum(n1,n2)
    nmin = int(n1/2)
    # print(nmin)
    
    for i in range(0,nmin):
        l1 = inds1[i]
        l2 = inds2[i]
        # print(l1,l2)
        model2.layers[l2].set_weights(model1.layers[l1].get_weights())
    
def get_corrupted_output(X, corruption_level=0.3):
    return np.random.binomial(1, 1-corruption_level, X.shape) * X

def get_gaussian_noise(X, sd=0.5):
    # Injecting small gaussian noise    
    X += np.random.normal(0, sd, X.shape)
    return X
    
# Create semi-supervised sets of labeled and unlabeled data
# where there are equal number of labels from each class
# 'x': MNIST images
# 'y': MNIST labels (binarized / 1-of-K coded)
def create_semisupervised(X, y, n_labeled, num_classes):
    

    [n, c, dim1, dim2] = X.shape
    Xr = X.reshape((n, c*dim1*dim2))

    # split by class
    def split_by_class(x, y, num_classes):
        result_x = [0]*num_classes
        result_y = [0]*num_classes

        Y = np_utils.to_categorical(y, num_classes)
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[idx_i,:]
            result_y[i] = Y[idx_i,:]
        return result_x, result_y

    
    x_list, y_list = split_by_class(Xr, y, num_classes)

    n_classes = y_list[0].shape[1]


    if n_labeled%n_classes != 0: 
        raise("n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")

    n_labels_per_class = n_labeled/n_classes
    x_labeled = [0]*n_classes
    x_unlabeled = [0]*n_classes
    y_labeled = [0]*n_classes
    y_unlabeled = [0]*n_classes

    for i in range(n_classes):
        idx = range(x_list[i].shape[0])
        random.shuffle(idx)
        x_labeled[i] = x_list[i][idx[:n_labels_per_class],:]
        y_labeled[i] = y_list[i][idx[:n_labels_per_class],:]
        x_unlabeled[i] = x_list[i][idx[n_labels_per_class:],:]
        y_unlabeled[i] = y_list[i][idx[n_labels_per_class:],:]

    X_l = np.vstack(x_labeled)
    X_l = X_l.reshape((X_l.shape[0], c, dim1, dim2))
    Y_l = np.vstack(y_labeled)

    X_u = np.vstack(x_unlabeled)
    X_u = X_u.reshape((X_u.shape[0], c, dim1, dim2))
    Y_u = np.vstack(y_unlabeled)

    return X_l, Y_l, X_u, Y_u

def remove_mean(X, scaler=None):
    [n,c,dim1,dim2] = X.shape
    Z = X.reshape(n,c*dim1*dim2)
    if scaler is None:
        scaler = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(Z)

    Z = scaler.transform(Z)
    Z = Z.reshape(n,c,dim1,dim2)

    return Z, scaler

def remove_mean_and_std(X, scaler=None):
    [n,c,dim1,dim2] = X.shape
    Z = X.reshape(n,c*dim1*dim2)
    if scaler is None:
        scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(Z)

    Z = scaler.transform(Z)
    Z = Z.reshape(n,c,dim1,dim2)

    return Z, scaler

def min_max(X, feature_range=(0,1)):
    scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
    [n,c,dim1,dim2] = X.shape

    Z = X.reshape(n,c*dim1*dim2)
    Z = scaler.fit_transform(Z)
    Z = Z.reshape(n,c,dim1,dim2)
    return Z


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


        

        
