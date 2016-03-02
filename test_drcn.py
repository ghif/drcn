import gzip
import cPickle as pickle
from sklearn import preprocessing


from loader import *
from models import *



src = 'svhn'
tgt = 'mnist'
model = 'drcn-st'
RESFILE = 'results/'+src+'-'+tgt+'_'+model+'_results_drop0.5_aug1_denoise0.5.pkl.gz'
PARAMFILE = 'results/'+src+'-'+tgt+'_'+model+'_weights_drop0.5_aug1_denoise0.5.pkl.gz'


print('Load data...')
inds = pickle.load(gzip.open('data_indices100.pkl.gz','rb'))


if src == 'svhn':
	(X_train, Y_train), (X_test, Y_test) = load_svhn()
	(_, _), (_, _), (X_tgt_test, Y_tgt_test) = load_mnist32x32()
	idx_src = inds['svhn_train']
	idx_tgt = inds['mnist_test']

elif src == 'mnist':
	if tgt == 'svhn':
		(X_train, Y_train), (_, _), (X_test, Y_test) = load_mnist32x32()
		(_, _), (X_tgt_test, Y_tgt_test) = load_svhn()
		idx_src = inds['mnist_train']
		idx_tgt = inds['svhn_test']
	else:
		(X_train, Y_train), (_, _), (X_test, Y_test) = load_mnist()
		(_, _), (X_tgt_test, Y_tgt_test) = load_usps()

elif src == 'usps':
	(X_train, Y_train), (X_test, Y_test) = load_usps()
	(_, _), (_, _), (X_tgt_test, Y_tgt_test) = load_mnist()

print('Preprocess data ...')
# X_train, scaler = remove_mean(X_train)
# X_test, _ = remove_mean(X_test, scaler=scaler)
# X_tgt_test, _ = remove_mean(X_tgt_test, scaler=scaler)
X_train = min_max(X_train)
X_test = min_max(X_test)
X_tgt_test = min_max(X_tgt_test)

[n, c, d1, d2] = X_train.shape

print('Load config and params...')
res = pickle.load(gzip.open(RESFILE,'rb'))

model = DRCN((c, d1, d2), 10, res['net_config'], res['ae_config'])
model.load_weights(PARAMFILE)


show_filter(X_train[idx_src], grayscale=True, filename='viz/'+src+'_'+tgt+'_'+model+'_X100-src-orig.png')
show_filter(model.convae_.predict(X_train[idx_src]), grayscale=True, filename='viz/'+src+'_'+tgt+'_'+model+'_X100-src-pred.png')
# show_filter(X_tgt_test[idx_tgt], grayscale=True, filename='viz/'+src+'_'+tgt+'_drcn-fixed_X100-tgt-orig.png')
# show_filter(model.convae_.predict(X_tgt_test[idx_tgt]), grayscale=True, filename='viz/'+src+'_'+tgt+'_drcn-fixed_X100-tgt-pred.png')