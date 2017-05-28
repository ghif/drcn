# Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)

This code is an implementation of the DRCN algorithm presented in [1].

[1] M. Ghifary, W. B. Kleijn, M. Zhang, D. Balduzzi, and W. Li. ["Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)"](https://arxiv.org/abs/1607.03516), European Conference on Computer Vision (ECCV), 2016

Contact:
```
Muhammad Ghifary (mghifary@gmail.com)
```

## Requirements
* Python 2.7
* Tensorflow-1.0.1
* Keras-2.0.0
* numpy
* h5py

## Usage
To run the experiment with the (grayscaled) SVHN dataset as the source domain and the MNIST dataset as the target domain
```
python main_sm.py
```

The core algorithm is implemented in __drcn.py__.
Data augmentation and denoising strategies are included as well.

## Results
The source to target reconstruction below (SVHN as the source) indicates the successful training of DRCN.

```
python reconstruct_images.py
```

![alt text](https://github.com/ghif/drcn/blob/master/rec_src.png "Source to Target Reconstruction")

The classification accuracies of one DRCN run are plotted as follows:

```
python plot_results.py
```

![alt text](https://github.com/ghif/drcn/blob/master/svhn-mnist_plot.png "Accuracy Plot")


