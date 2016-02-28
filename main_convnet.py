from sklearn import preprocessing

from loader import *
from models import * # contains Lasagne


# hyperparameters

# hyperparameters
learning_rate = 1e-4
batch_size = 128
nb_epoch = 50
augmentation = True
dropout = 0.5
dense_dim = 300


src = 'svhn'
tgt = 'mnist'

RESFILE = 'results/'+src+'-'+tgt+'_convnet_results_drop%.1f_aug%d_h%d.pkl.gz' % (dropout, augmentation, dense_dim)
PARAMFILE = 'results/'+src+'-'+tgt+'_convnet_weights_drop%.1f_aug%d_h%d.pkl.gz' % (dropout, augmentation, dense_dim)
print(PARAMFILE)

# Load data
if src == 'svhn':
	(X_train, Y_train), (X_test, Y_test) = load_svhn()
	(_, _), (_, _), (X_tgt_test, Y_tgt_test) = load_mnist32x32()
elif src == 'mnist':
	(X_train, Y_train), (_, _), (X_test, Y_test) = load_mnist32x32()
	(_, _), (X_tgt_test, Y_tgt_test) = load_svhn()


print('Preprocess data ...')
# X_train, scaler = remove_mean(X_train)
# X_test, _ = remove_mean(X_test, scaler=scaler)
# X_tgt_test, _ = remove_mean(X_tgt_test, scaler=scaler)

X_train = min_max(X_train)
X_test = min_max(X_test)
X_tgt_test = min_max(X_tgt_test)

[n, c, d1, d2] = X_train.shape


##### CONVEA ####
model = ConvNet()
model.create_architecture(input_shape=(c, d1, d2), dense_dim=dense_dim, dout=10, dropout=dropout)
model.compile(lr=learning_rate, loss_function='categorical_crossentropy')
model.train(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,  shuffle=True, augmentation=False,
	X_test=X_test, Y_test=Y_test, X_tgt=X_tgt_test, Y_tgt=Y_tgt_test,
	RESFILE=RESFILE, PARAMFILE=PARAMFILE)