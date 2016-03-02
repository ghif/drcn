import cPickle as pickle
import numpy as np
import gzip

src = 'svhn'
tgt = 'mnist'
datapair = src+'-'+tgt
# model = 'convnet'
model = 'drcn-st'
dropout_rate = 0.5
is_aug = 1
denoising = 0.5

RESFILE = 'results/'+datapair+'_'+model+'_results_drop%.1f_aug%d_denoise%.1f.pkl.gz' % (dropout_rate, is_aug, denoising)
# RESFILE = 'results/'+datapair+'_'+model+'_results_drop%.1f_aug%d_h300.pkl.gz' % (dropout_rate, is_aug)

print(RESFILE)
res = pickle.load(gzip.open(RESFILE,'rb'))


print('Train accs : (%.4f, %.4f)' % (res['accs'][-1], np.max(res['accs'])))
print('Test accs : (%.4f, %.4f)' % (res['test_accs'][-1], np.max(res['test_accs'])))
print('Target test accs : (%.4f, %.4f)' % (res['tgt_accs'][-1], np.max(res['tgt_accs'])))


# print('Train accs : (%.2f, %.2f)' % (res['train_accs'][-1], np.max(res['train_accs'])))
# print('Test accs : (%.2f, %.2f)' % (res['test_accs'][-1], np.max(res['test_accs'])))
# print('Target test accs : (%.2f, %.2f)' % (res['tgt_test_accs'][-1], np.max(res['tgt_test_accs'])))
