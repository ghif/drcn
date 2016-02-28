
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# import matplotlib.gridspec as gridspec


def plot_images(imgs,  filename=None,rescale=True, add_mean=False, grayscale=False):
    num_examples = imgs.shape[0]
    # print(num_examples)
    s = np.sqrt(num_examples).astype('uint8')
    # print('s = ',s)

    fig = plt.figure()
    # gsl = gridspec.GridSpec(s,s)
    # gsl.update(wspace=0.025, hspace=0.05) # set the spacing between axes
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in xrange(num_examples):
        plt.subplot(s, s, i+1, aspect='equal')
        # plt.subplot(s, s, gsl[i])
        img = imgs[i,:,:,:]
        show_image(img, rescale=rescale, add_mean=add_mean, grayscale=grayscale)

    if filename is not None:
        plt.savefig(filename)

    # plt.tight_layout()
    plt.show()

# def show_image2():
def show_image(img, rescale=True, add_mean=False, grayscale=False):
    """
    Utility to show an image. In our ConvNets, images are 3D slices of 4D
    volumes; to visualize them we need to squeeze out the extra dimension,
    flip the axes so that channels are last, add the mean image, convert to
    uint8, and possibly rescale to be between 0 and 255. To make figures
    prettier we also need to suppress the axis labels after imshow.
    
    Input:
    - img: (1, C, H, W) or (C, H, W) or (1, H, W) or (H, W) giving
      pixel data for an image.
    - rescale: If true rescale the data to fit between 0 and 255
    - add_mean: If true add the training data mean image
    """
    cmap = None
    if grayscale == True:
        cmap = cm.Greys_r


    img = img.copy()
    if add_mean:
        img += mean_img
    img = img.squeeze()
    # print('show_images squeeze',img.shape)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    if rescale:
        low, high = np.min(img), np.max(img)
        img = 255.0 * (img - low) / (high - low)
    # print('show_images final',img.shape)
    plt.imshow(img.astype('uint8'), cmap=cmap)
    plt.gca().axis('off')

def show_filter(X, padsize=1, padval=0, grayscale=False, filename=None, conv=True):
    data = np.copy(X)
    if conv:
        [n, c, d1, d2] = data.shape
        if c == 1:
            data = data.reshape((n, d1, d2))
        else:
            data = data.transpose(0,2,3,1)
    else:
        # print(data.shape)
        [n, d] = data.shape
        s = int(np.sqrt(d))
        data = data.reshape((n, s, s))

    vis_square(data, padsize=padsize, padval=padval, grayscale=grayscale, filename=filename)

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, grayscale=False, filename=None):

    # print('min : ', np.min(data))
    # print('max : ', np.max(data))

    # this is not needed !
    data -= data.min()
    data /= data.max()

    # print('min : ', np.min(data))
    # print('max : ', np.max(data))

    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    
    
    if grayscale == True:
        plt.imshow(data, cmap=cm.Greys_r)
    else:
        plt.imshow(data)

    plt.axis('off')

    if filename is None:
        plt.draw()
        plt.show()
        
    else:
        plt.savefig(filename, format='png')


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None, filename=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(digits.data.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def plot_accs(dict_ac1, dict_ac2,
    max_x = None,
    img_file='fig/svhn-mnist_accs_plot.png'
    ):
    
    # nepoch = len(dict_ac['train_accs'])
    if max_x is None:
        max_x = len(dict_ac1['train_accs']) #total epoch
    x = range(0, max_x)

    # train accs
    y = dict_ac1['train_accs']
    y = [i * 100 for i in y]
    y = y[0:max_x]
    line1_train, = plt.plot(x, y, color='blue',linewidth=2, linestyle=':')

    y = dict_ac2['train_accs']
    y = [i * 100 for i in y]
    y = y[0:max_x]
    line2_train, = plt.plot(x, y, color='blue',linewidth=2)

    # tgt accs
    y = dict_ac1['tgt_accs']
    y = [i * 100 for i in y]
    y = y[0:max_x]
    line1_tgt, = plt.plot(x, y, color='red',linewidth=2, linestyle=':')

    y = dict_ac2['tgt_accs']
    y = [i * 100 for i in y]
    y = y[0:max_x]
    line2_tgt, = plt.plot(x, y, color='red',linewidth=2)

    plt.legend([line1_train, line2_train, line1_tgt, line2_tgt],
        ['ConvNet (source accuracy)', 'MDGN (source accuracy)', 'Convnet (target accuracy)','MDGN (target accuracy)'],
        loc=4
    )


    # plt.title('')
    plt.xlabel('# of epoch')
    plt.ylabel('Accuracy (%)')

    plt.grid() #grid on
    axisrange=[0, max_x, 0, 104]
    plt.axis(axisrange)

    # set font
    font = {'weight' : 'bold','size'   : 14}

    matplotlib.rc('font',**font)
    plt.savefig(img_file)
    plt.show()


