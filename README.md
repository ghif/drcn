# Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)

This code is an implementation of the DRCN algorithm presented in [1].

[1] M. Ghifary, W. B. Kleijn, M. Zhang, D. Balduzzi, and W. Li. ["Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)"](https://arxiv.org/abs/1607.03516), European Conference on Computer Vision (ECCV), 2016

Contact:
```
Muhammad Ghifary (mghifary@gmail.com)
```

## Requirements
* Python 2.7
* Tensorflow-0.10.0
* Keras-1.2.0
* numpy
* h5py

## Usage
To run the experiment with SVHN as the source domain and MNIST as the target domain
```
python main_sm.py
```

The core algorithm is implemented in drcn.py.


## TO DO
* Data augmentation and denoising

