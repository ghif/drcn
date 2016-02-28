import gzip
import cPickle as pickle



from loader import *
from models import *



src = 'svhn'
tgt = 'mnist'

RESFILE = 'results/'+src+'-'+tgt+'_convae_results_denoising0.pkl.gz'
PARAMFILE = 'results/'+src+'-'+tgt+'_convae_weights_denoising0.pkl.gz'


print('Load data...')
inds = pickle.load(gzip.open('data_indices100.pkl.gz','rb'))

if src == 'svhn':
	(X_train, Y_train), (X_test, Y_test) = load_svhn()
	(_, _), (_, _), (X_tgt_test, Y_tgt_test) = load_mnist32x32()
	idx_src = inds['svhn_train']
	idx_tgt = inds['mnist_test']
elif src == 'mnist':
	(X_train, Y_train), (_, _), (X_test, Y_test) = load_mnist32x32()
	(_, _), (X_tgt_test, Y_tgt_test) = load_svhn()
	idx_src = inds['mnist_train']
	idx_tgt = inds['svhn_test']

[n, c, d1, d2] = X_train.shape

print('Load config and params...')
res = pickle.load(gzip.open(RESFILE,'rb'))

model = ConvAE()
model.create_architecture(input_shape=(c, d1, d2))
model.compile(lr=res['lr'], loss_function='squared_error')
model.load_weights(PARAMFILE)

show_filter(X_train[idx_src], grayscale=True, filename='viz/'+src+'_'+tgt+'_convae_X100-src-orig.png')
show_filter(model.predict(X_train[idx_src]), grayscale=True, filename='viz/'+src+'_'+tgt+'_convae_X100-src-pred.png')
show_filter(X_tgt_test[idx_tgt], grayscale=True, filename='viz/'+src+'_'+tgt+'_convae_X100-tgt-orig.png')
show_filter(model.predict(X_tgt_test[idx_tgt]), grayscale=True, filename='viz/'+src+'_'+tgt+'_convae_X100-tgt-pred.png')