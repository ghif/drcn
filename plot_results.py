import matplotlib.pyplot as plt
import numpy as np

hist = np.load('svhn-mnist_drcn_v2_hist.npy').item()

x = range(len(hist['accs']))

y1 = np.array(hist['accs']) * 100.
y2 = np.array(hist['val_accs']) * 100.
y3 = np.array(hist['test_accs']) * 100.


plt.plot(x, y1, label='Train Acc. (SVHN)')
plt.plot(x, y2, label='Validation Acc. (SVHN)')
plt.plot(x, y3, label='Test Acc. (MNIST)')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.legend(loc=4)
plt.grid(True)

plt.tight_layout()

plt.savefig('svhn-mnist_plot.png')
