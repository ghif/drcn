from sklearn import preprocessing

from loader import *
from models import * # contains Lasagne


# hyperparameters

# hyperparameters
learning_rate = 1e-4
batch_size = 32
nb_epoch = 50
denoising = False


src = 'mnist'
tgt = 'svhn'

RESFILE = 'results/'+src+'-'+tgt+'_convae_results_denoising%d.pkl.gz' % (denoising)
PARAMFILE = 'results/'+src+'-'+tgt+'_convae_weights_denoising%d.pkl.gz' % (denoising)
PREDICTPREFIX = src+'-'+tgt+'_convae_'
print(RESFILE)	


# Load data
if src == 'svhn':
	(X_train, Y_train), (X_test, Y_test) = load_svhn()
	(_, _), (_, _), (X_tgt_test, Y_tgt_Test) = load_mnist32x32()
elif src == 'mnist':
	(X_train, Y_train), (_, _), (X_test, Y_test) = load_mnist32x32()
	(_, _), (X_tgt_test, Y_tgt_test) = load_svhn()


print('Preprocess data ...')
# X_train, scaler = remove_mean(X_train)
# X_test, _ = remove_mean(X_test, scaler=scaler)
# X_tgt_test, _ = remove_mean(X_tgt_test, scaler=scaler)

# X_train = min_max(X_train)
# X_test = min_max(X_test)
# X_tgt_test = min_max(X_tgt_test)

[n, c, d1, d2] = X_train.shape


##### CONVEA ####
model = ConvAE()
model.create_architecture(input_shape=(c, d1, d2))
model.compile(lr=learning_rate, loss_function='squared_error')
model.train(X_tgt_test, batch_size=batch_size, nb_epoch=nb_epoch, 
	denoising=denoising,
	X_valid=X_train,
	RESFILE=RESFILE, PARAMFILE=PARAMFILE, PREDICTPREFIX=PREDICTPREFIX)