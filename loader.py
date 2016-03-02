import os
import cPickle as pickle
import gzip

import theano
import theano.tensor as T
import numpy as np
import sys
import scipy.io as sio
from sklearn import preprocessing

def load_office31_decaf6(domain='amazon'):
    domdir = '/u/students/gif/Desktop/PhD/Tutorial/dataset/office/' #linux
    dompath = domdir + domain+'_decaf6_31.pkl.gz'
    print(dompath)
    f = gzip.open(dompath)
    M = pickle.load(f)
    f.close()
    
    X = M['data']
    y = M['labels']
    return (X,y)

def load_office_raw(src='amazon',tgt='webcam'):
    # dirname='I:\Data\PhD Life\Tutorial\dataset\office\domain_adaptation_images' #windows
    dirname='/u/students/gif/Desktop/PhD/Tutorial/dataset/office/domain_adaptation_images' #windows
    srcpath = os.path.join(dirname,src+'224x224.pkl')
    tgtpath = os.path.join(dirname,tgt+'224x224.pkl')
    
    srcdic = pickle.load(open(srcpath,'rb'))


    X_train = srcdic["data"]
    y_train = srcdic["labels"]

    tgtdic = pickle.load(open(tgtpath,'rb'))
    X_test = tgtdic["data"]
    y_test = tgtdic["labels"]

    return (X_train, y_train), (X_test, y_test)

def load_cifar10(mode=0):
    dirname='/u/students/gif/Desktop/PhD/Tutorial/dataset/cifar-10-batches-py/'
    nb_test_samples = 10000
    nb_train_samples = 50000

    # Load training
    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1,6):
        train_batch = 'data_batch_'+str(i)
        train_path = dirname  + train_batch
        
        f = open(train_path,'rb')
        train_dict = pickle.load(f)
        f.close()

        data = train_dict['data']
        labels = train_dict['labels']
        X_train[(i-1)*10000:i*10000, :, :, :] = data.reshape(data.shape[0],3,32,32)
        y_train[(i-1)*10000:i*10000] = labels

    # Load test
    test_path = dirname + 'test_batch'
    f = open(test_path, 'rb')
    test_dict = pickle.load(f)
    f.close()

    X_test = np.zeros((nb_test_samples, 3, 32, 32), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")

    data = test_dict['data']
    labels = test_dict['labels']
    X_test[:,:,:,:] = data.reshape(data.shape[0],3,32,32)
    y_test[:] = labels

    # take only 8 shared classes with STL10
    classes = [0,2,3,4,5,7,8,9]
    X_train, y_train = get_some_classes(X_train, y_train, classes)
    X_test, y_test = get_some_classes(X_test, y_test, classes)

    
    return (X_train, y_train), (X_test, y_test)


def load_stl10():
    dirname = '/u/students/gif/Desktop/PhD/Tutorial/dataset/STL-10/'
    filename = 'stl10.pkl.gz'
    fpath = os.path.join(dirname,filename)
    (X_train, y_train), (X_test, y_test) = pickle.load(gzip.open(fpath,'rb'))

    # take only 8 shared classes with CIFAR10
    classes = [0,1,3,4,5,6,8,9]
    X_train, y_train = get_some_classes(X_train, y_train, classes)
    X_test, y_test = get_some_classes(X_test, y_test, classes)

    return (X_train, y_train), (X_test, y_test)

def get_some_classes(X,y,classes):
    G_list = []
    L_list = []

    idx = 0
    for c in classes:
        inds_c = np.where(y == c)
        inds_c = inds_c[0]

        G = X[inds_c]
        L = np.zeros(y[inds_c].shape) + idx

        G_list.append(G)
        L_list.append(L)


        idx += 1

    X_new = G_list[0]
    y_new = L_list[0]
    for c in range(1,len(G_list)):
        X_new = np.concatenate((X_new, G_list[c]), axis=0)
        y_new = np.concatenate((y_new, L_list[c]), axis=0)

    return X_new, y_new


def load_mnist_raw():
    # dirname = 'I:\Data\PhD Life\Tutorial\dataset\MNIST' #windows
    dirname = '/u/students/gif/Desktop/PhD/Tutorial/dataset/MNIST' #linux
    filename = 'mnist.pkl.gz'
    fpath = os.path.join(dirname,filename)
    f = gzip.open(fpath,'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding="bytes")
    f.close()
    return data # (X_train, y_train), (X_test, y_test)
    
def load_mnist(mode=0):
    # dataset='I:\Data\PhD Life\Tutorial\Python\data\mnist.pkl.gz' # linux
    # dataset='/u/students/gif/Desktop/PhD/Tutorial/Python/data/mnist.pkl.gz' # linux
    dataset='/u/students/gif/Desktop/PhD/Tutorial/dataset/MNIST/mnist.pkl.gz' # linux
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    
    if mode == 0:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set


    else:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    
    
    train_set_x = train_set_x * 255.0
    valid_set_x = valid_set_x * 255.0
    test_set_x = test_set_x *255.0



    train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, 28, 28).astype('float32')
    valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], 1, 28, 28).astype('float32')
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, 28, 28).astype('float32')
    train_set_y = train_set_y.astype('uint8')
    valid_set_y = valid_set_y.astype('uint8')
    test_set_y = test_set_y.astype('uint8')

    rval = (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)
    return rval

def load_mnist32x32(mode=0):
    # dataset='I:\Data\PhD Life\Tutorial\Python\data\mnist.pkl.gz' # linux
    # dataset='/u/students/gif/Desktop/PhD/Tutorial/dataset/MNIST/mnist32x32.pkl.gz' # linux
    dataset = '/local/scratch/gif/dataset/MNIST/mnist32x32.pkl.gz' #the-villa
    # dataset = 'I:\Data\PhD Life\Tutorial\dataset\MNIST\mnist32x32.pkl.gz' # laptop
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    
    if mode == 0:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set


    else:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    
    # print(np.max(train_set_x))
    # print(np.min(train_set_x))
    
    # train_set_x = train_set_x
    # valid_set_x = valid_set_x
    # test_set_x = test_set_x


    train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, 32, 32)
    valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], 1, 32, 32)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, 32, 32)

    return (train_set_x.astype('float32'), train_set_y.astype('uint8')), (valid_set_x.astype('float32'), valid_set_y.astype('uint8')), (test_set_x.astype('float32'), test_set_y.astype('uint8'))

def load_usps(mode=0):
    # dataset = 'I:\Data\PhD Life\Tutorial\dataset\USPS\usps.pkl.gz' #windows
    dataset='/u/students/gif/Desktop/PhD/Tutorial/dataset/USPS/usps.pkl.gz' # linux
    f = gzip.open(dataset,'rb')
    train_set, test_set = pickle.load(f)
    f.close()
    
    if mode == 0:
        train_set_x, train_set_y = train_set
        test_set_x, test_set_y = test_set
    else:
        train_set_x, train_set_y = shared_dataset(train_set)
        test_set_x, test_set_y = shared_dataset(test_set)


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

def load_mnist256(dataset):
    f = gzip.open(dataset,'rb')
    domains = pickle.load(f)
    f.close()
    return domains


def construct_src_tar_from_mnist256(left_out_idx=0,mode=0,dataset='data/MNIST256_6rotations.pkl.gz'):
    domains = load_mnist256(dataset)
    dom_idx = range(0,len(domains))
    del dom_idx[left_out_idx]

    n_train_list = []

    for i in dom_idx:
        data_x, data_y = domains[i]
        n_train_list.append(data_x.shape[0])

    n_train = sum(n_train_list)

    d = domains[left_out_idx][0].shape[1] #data dimension

    src_domains = domains[:] # clone the list

    del src_domains[left_out_idx]

    if mode == 0:
        x_train = numpy.zeros(shape=(n_train,d))
        y_train = numpy.zeros(shape=(n_train,))

        # fill the values
        i1 = 0

        for i in range(0,len(n_train_list)):
            i2 = i1+n_train_list[i]
            x_train[i1:i2] = src_domains[i][0]
            y_train[i1:i2] = src_domains[i][1]
            i1 = i2

        x_test = domains[left_out_idx][0]
        y_test = domains[left_out_idx][1]

        train_set = (x_train, y_train)
        test_set = (x_test, y_test)

        train_set_x, train_set_y = shared_dataset(train_set)
        test_set_x, test_set_y = shared_dataset(test_set)

        return [(train_set_x, train_set_y),(test_set_x, test_set_y)]

    else: # for MTNN
        n_src = len(src_domains)
       
        rval = []
        for d in range(0,n_src):
            train_set_x, train_set_y = shared_dataset(src_domains[d])
            rval += [(train_set_x, train_set_y)]

        test_set_x, test_set_y = shared_dataset(domains[left_out_idx])
        rval += [(test_set_x, test_set_y)]
        return rval

def construct_mnistrot_for_mtae(left_out_idx=0, balanced=True,
    dataset='data/MNIST256_6rotations.pkl.gz'
    ):

    domains = load_mnist256(dataset=dataset)
    dom_idx = range(0,len(domains))
    del dom_idx[left_out_idx]

    n_train_list = []

    for i in dom_idx:
        data_x, data_y = domains[i]
        n_train_list.append(data_x.shape[0])

    n_train = sum(n_train_list)

    

    src_domains = domains[:] # clone the list

    del src_domains[left_out_idx]

    

    # create duplicated data for each domain
    rval = []
    n_src_dom = len(n_train_list)
    for d in range(0,n_src_dom):
        xd_train = numpy.tile(src_domains[d][0],(n_src_dom,1))
        yd_train = numpy.tile(src_domains[d][1],(n_src_dom,1))
        
        train_set_dup = (xd_train, yd_train)
        train_x_d, train_y_d = shared_dataset(train_set_dup)
        rval += [train_x_d]


    d = domains[left_out_idx][0].shape[1] #data dimension
    x_train = numpy.zeros(shape=(n_train,d))
    y_train = numpy.zeros(shape=(n_train,))

    # create data combining all domains
    i1 = 0
    for i in range(0,n_src_dom):
        i2 = i1+n_train_list[i]
        x_train[i1:i2] = src_domains[i][0]
        y_train[i1:i2] = src_domains[i][1]
        i1 = i2

    train_set = (x_train, y_train)

    train_set_x, train_set_y = shared_dataset(train_set)

    rval += [train_set_x]
    return rval
    
def load_vlcs(left_out_idx=0,mode=0):
    train_domains, test_domains = VlcsToPickle()

    dom_idx = range(0,len(train_domains))
    del dom_idx[left_out_idx]

    n_train_list = []

    for i in dom_idx:
        data_x, data_y = train_domains[i]
        n_train_list.append(data_x.shape[0])

    n_train = sum(n_train_list)

    d = train_domains[left_out_idx][0].shape[1] #data dimension
    
    src_domains = train_domains[:] # clone the list
    del src_domains[left_out_idx] # remove the left-out domain

    if mode == 0: # for all standard AE models and SVM classifier
        x_train = numpy.zeros(shape=(n_train,d))
        y_train = numpy.zeros(shape=(n_train,))

        # fill the values
        i1 = 0

        for i in range(0,len(n_train_list)):
            i2 = i1+n_train_list[i]
            x_train[i1:i2] = src_domains[i][0]
            y_train[i1:i2] = src_domains[i][1]
            i1 = i2

        x_test = test_domains[left_out_idx][0]
        y_test = test_domains[left_out_idx][1]

        train_set = (x_train, y_train)
        test_set = (x_test, y_test)

        train_set_x, train_set_y = shared_dataset(train_set)
        test_set_x, test_set_y = shared_dataset(test_set)

        return [(train_set_x, train_set_y),(test_set_x, test_set_y)]
    elif mode == 1: #for MTAE

        # create duplicated data for each domain
        rval = []
        n_src_dom = len(n_train_list)
        for d in range(0,n_src_dom):
            xd_train = numpy.tile(src_domains[d][0],(n_src_dom,1))
            yd_train = numpy.tile(src_domains[d][1],(n_src_dom,1))
            
            train_set_dup = (xd_train, yd_train)
            train_x_d, train_y_d = shared_dataset(train_set_dup)
            rval += [train_x_d]


        d = train_domains[left_out_idx][0].shape[1] #data dimension
        x_train = numpy.zeros(shape=(n_train,d))
        y_train = numpy.zeros(shape=(n_train,))

        # create data combining all domains
        i1 = 0
        for i in range(0,n_src_dom):
            i2 = i1+n_train_list[i]
            x_train[i1:i2] = src_domains[i][0]
            y_train[i1:i2] = src_domains[i][1]
            i1 = i2

        train_set = (x_train, y_train)

        train_set_x, train_set_y = shared_dataset(train_set)

        rval += [train_set_x]
        return rval

def load_office(left_out_inds=[0,1], mode=0):
    domains = OfficeToPickle()
    dim = domains[0][0].shape[1] # data dimension
    

    n_dom = len(domains) # number of domains

    src_domains = []
    tgt_domains = []

    n_train_list = []
    n_test_list = []
    for idx in range(0,n_dom):
        if idx in left_out_inds:
            tgt_domains.append(domains[idx])
            n_test_list.append(domains[idx][0].shape[0])
        else:
            src_domains.append(domains[idx])
            n_train_list.append(domains[idx][0].shape[0])
    

    n_train = sum(n_train_list)
    n_test = sum(n_test_list)
    

    if mode == 0: # for all standard AE models and SVM classifier
        x_train = numpy.zeros(shape=(n_train,dim))
        y_train = numpy.zeros(shape=(n_train,))

        # fill the training data values
        i1 = 0
        for i in range(0,len(n_train_list)):
            i2 = i1+n_train_list[i]
            x_train[i1:i2] = src_domains[i][0]
            y_train[i1:i2] = src_domains[i][1]
            i1 = i2

        x_test = numpy.zeros(shape=(n_test,dim))
        y_test = numpy.zeros(shape=(n_test,))

        i1 = 0
        for i in range(0,len(n_test_list)):
            i2 = i1+n_test_list[i]
            x_test[i1:i2] = tgt_domains[i][0]
            y_test[i1:i2] = tgt_domains[i][1]
            i1 = i2

        train_set = (x_train, y_train)
        test_set = (x_test, y_test)

        train_set_x, train_set_y = shared_dataset(train_set)
        test_set_x, test_set_y = shared_dataset(test_set)

        return [(train_set_x, train_set_y),(test_set_x, test_set_y)]
    elif mode == 1: #for MTAE
        # create duplicated data for each domain
        
        rval = []
        n_src_dom = len(n_train_list)
        for d in range(0,n_src_dom):
            xd_train = numpy.tile(src_domains[d][0],(n_src_dom,1))
            yd_train = numpy.tile(src_domains[d][1],(n_src_dom,1))
            
            # nd = n_train_list[d]
            
            # xd_train = numpy.zeros(shape=(n_train,dim))
            # yd_train = numpy.zeros(shape=(n_train,))

            # i1 = 0
            # for j in range(0,n_src_dom):
            #     if d != j:
                    
            #     else:
            #         i2 = i1+n_train_list[j]
            #         xd_train[i1:i2] = src_domains[d][0]



            train_set_dup = (xd_train, yd_train)
            train_x_d, train_y_d = shared_dataset(train_set_dup)
            rval += [train_x_d]


        
        x_train = numpy.zeros(shape=(n_train,dim))
        y_train = numpy.zeros(shape=(n_train,))

        # create data combining all domains
        i1 = 0
        for i in range(0,n_src_dom):
            i2 = i1+n_train_list[i]
            x_train[i1:i2] = src_domains[i][0]
            y_train[i1:i2] = src_domains[i][1]
            i1 = i2

        train_set = (x_train, y_train)

        train_set_x, train_set_y = shared_dataset(train_set)

        rval += [train_set_x]
        return rval



def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def VlcsToPickle():
    dl = ['VOC2007','LabelMe','Caltech101','SUN09']
    ndom = len(dl)

    train_domains = []
    test_domains = []
    for d in range(0,ndom):
        # load mat dataset
        dom = dl[d]
        mat = sio.loadmat('data/'+dom+'_neuron_decaf6_split')

        Xtrain = numpy.array(mat['Xtrain']).astype('float32')
        n = Xtrain.shape[0]
        Ytrain = numpy.array(mat['Ytrain']).astype('int64').reshape(n,) - 1
        train_domains.append((Xtrain,Ytrain))
        
        Xtest = numpy.array(mat['Xtest']).astype('float32')
        n = Xtest.shape[0]
        Ytest = numpy.array(mat['Ytest']).astype('int64').reshape(n,) - 1
        test_domains.append((Xtest,Ytest))


    return train_domains, test_domains

def OfficeToPickle():
    dl = ['Amazon','Webcam','Dslr','Caltech']
    ndom = len(dl)

    domains = []
    for d in range(0,ndom):
        dom = dl[d]
        mat = sio.loadmat('data/'+dom+'_DeCAF6')

        X = numpy.array(mat['X']).astype('float32')
        # print numpy.max(X)
        n = X.shape[0]
        Y = numpy.array(mat['Y']).astype('int64').reshape(n,)
        Y = Y - 1

        domains.append((X,Y))

    return domains

def convertMATtoPickle(dataset='MNISTr'):
    domains = []
    if dataset == 'MNISTr':
        max_val = 255.0
    
        mat = sio.loadmat('data/MNIST1000_basic_16x16')
        X0 = numpy.array(mat['X']).astype('float32')/max_val
        n = X0.shape[0]
        Y0 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X0, Y0))

        mat = sio.loadmat('data/MNIST1000_rotated_-15_16x16')
        X15 = numpy.array(mat['X']).astype('float32')/max_val
        n = X15.shape[0]
        Y15 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X15, Y15))

        mat = sio.loadmat('data/MNIST1000_rotated_-30_16x16')
        X30 = numpy.array(mat['X']).astype('float32')/max_val
        n = X30.shape[0]
        Y30 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X30, Y30))

        mat = sio.loadmat('data/MNIST1000_rotated_-45_16x16')
        X45 = numpy.array(mat['X']).astype('float32')/max_val
        n = X45.shape[0]
        Y45 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X45, Y45))

        mat = sio.loadmat('data/MNIST1000_rotated_-60_16x16')
        X60 = numpy.array(mat['X']).astype('float32')/max_val
        n = X60.shape[0]
        Y60 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X60, Y60))

        mat = sio.loadmat('data/MNIST1000_rotated_-75_16x16')
        X75 = numpy.array(mat['X']).astype('float32')/max_val
        n = X75.shape[0]
        Y75 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X75, Y75))

        filename = 'MNIST256_6rotations.pkl'

    elif dataset == 'MNISTs':

        max_val = 255.0

        mat = sio.loadmat('data/MNIST1000_basic_16x16')
        X0 = numpy.array(mat['X']).astype('float32')/max_val
        n = X0.shape[0]
        Y0 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X0, Y0))

        mat = sio.loadmat('data/MNIST1000_scaled_09_16x16')
        X09 = numpy.array(mat['X']).astype('float32')/max_val
        n = X09.shape[0]
        Y09 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X09, Y09))

        mat = sio.loadmat('data/MNIST1000_scaled_08_16x16')
        X08 = numpy.array(mat['X']).astype('float32')/max_val
        n = X08.shape[0]
        Y08 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X08, Y08))


        mat = sio.loadmat('data/MNIST1000_scaled_07_16x16')
        X07 = numpy.array(mat['X']).astype('float32')/max_val
        n = X07.shape[0]
        Y07 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X07, Y07))

        mat = sio.loadmat('data/MNIST1000_scaled_06_16x16')
        X06 = numpy.array(mat['X']).astype('float32')/max_val
        n = X06.shape[0]
        Y06 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X06, Y06))

        filename = 'MNIST256_5scales.pkl'

    elif dataset == 'ETH80p':

        max_val = 255.0

        mat = sio.loadmat('data/ETH80p_0')
        X0 = numpy.array(mat['X']).astype('float32')/max_val
        n = X0.shape[0]
        Y0 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X0, Y0))

        mat = sio.loadmat('data/ETH80p_22')
        X22 = numpy.array(mat['X']).astype('float32')/max_val
        n = X22.shape[0]
        Y22 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X22, Y22))

        mat = sio.loadmat('data/ETH80p_45')
        X45 = numpy.array(mat['X']).astype('float32')/max_val
        n = X45.shape[0]
        Y45 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X45, Y45))


        mat = sio.loadmat('data/ETH80p_68')
        X68 = numpy.array(mat['X']).astype('float32')/max_val
        n = X68.shape[0]
        Y68 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X68, Y68))

        mat = sio.loadmat('data/ETH80p_90')
        X90 = numpy.array(mat['X']).astype('float32')/max_val
        n = X90.shape[0]
        Y90 = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X90, Y90))

        filename = 'ETH80_5pitch.pkl'

    elif dataset == 'ETH80y':

        max_val = 255.0

        mat = sio.loadmat('data/ETH80y_0')
        X = numpy.array(mat['X']).astype('float32')/max_val
        n = X.shape[0]
        Y = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X, Y))

        mat = sio.loadmat('data/ETH80y_45')
        X = numpy.array(mat['X']).astype('float32')/max_val
        n = X.shape[0]
        Y = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X, Y))

        mat = sio.loadmat('data/ETH80y_90')
        X = numpy.array(mat['X']).astype('float32')/max_val
        n = X.shape[0]
        Y = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X, Y))


        mat = sio.loadmat('data/ETH80y_135')
        X = numpy.array(mat['X']).astype('float32')/max_val
        n = X.shape[0]
        Y = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X, Y))

        mat = sio.loadmat('data/ETH80y_180')
        X = numpy.array(mat['X']).astype('float32')/max_val
        n = X.shape[0]
        Y = numpy.array(mat['Y']).astype('int64').reshape(n,)
        domains.append((X, Y))

        filename = 'ETH80_5yaw.pkl'


        

    pickle.dump(domains, open("data/"+filename,"wb"))
