from __future__ import print_function

import six.moves.cPickle as pickle
from six.moves import urllib

import gzip
import os
import theano
import theano.tensor as T

from PIL import Image
import numpy as np

def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if data_file == 'mnist.pkl.gz':
        train_set, valid_set, test_set = load_MNIST(dataset)
    elif data_file == 'vanhateren':
        train_set, valid_set, test_set = load_vanhateren(dataset)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_vanhateren(dataset):
    data_dir, data_file = os.path.split(dataset)
    assert(data_file == 'vanhateren')

    # alternate data_dir not currently working, so just use this one
    data_dir = '../data/vanhateren/'
    nIms = 20 # TODO: specified in the paper as 2000
    filename_spec = 'imk%.5d.iml'

    # check if the highest-numbered file exists
    max_file = os.path.join(data_dir, filename_spec % (nIms-1))
    if not os.path.isfile(max_file):
        origin = (
        'http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/imc/'
        ) 
        
        print('Downloading data from %s' % origin)
        os.chdir(data_dir)
        for i in range(nIms):
            filename = filename_spec % i
            urllib.request.urlretrieve(origin, filename)
        os.chdir('../../code')

        print('Done downloading')

    patchsz = 32
    which_ims = range(nIms)
    n_patches = 1000 # TODO: specified in the paper as 10,000
    datasz = [1024, 1536]
    datatype = 'uint16'

    def read_image(dataloc):
        # datasz and datatype are inherited from context
        readfile = open(dataloc, 'rb').read()
        im_array = np.fromstring(readfile, dtype=datatype).byteswap() # ????? RESTART HERE why isn't this the right size?
        im_array = im_array.reshape(datasz)
        im_array = im_array.astype('float32')
        return im_array
        # NOTE: this format can be used with
        # im = Image.fromarray(im_array) if desired

    all_files = [os.path.join(data_dir, filename_spec % i) for i in which_ims]

    import pdb; pdb.set_trace()
    ims = map(read_image, all_files)

    # TODO: Load the patches
    patches = make_patches(data_dir, patchsz, which_ims, n_patches)

    # TODO: make train/test/etc

def load_MNIST(dataset):
    data_dir, data_file = os.path.split(dataset)
    assert(data_file == 'mnist.pkl.gz')

    # Download if needed
    if (not os.path.isfile(dataset)):
        origin = (
        'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)
        print('Done downloading')

    # Load the dataset
    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)


def make_patches(data_dir, patchsz, which_ims, n_patches):
    return 1

#patchsz = 32;#
#x_starts = range(0, datasz[0], patchsz)
#y_starts = range(0, datasz[1], patchsz)
#
## Paper result:
## "We used 100,000 14-by-14 image patches randomly sampled from an #ensemble of 2000 images; each subset of 200 patches was used as a #mini-batch."
#
## TODO: make this awesome
#images = np.asarray(map(self.read_image, items))
#
## TODO: then use this
#rval4 = random_patches(images[:, :, :, None], N, prows, pcols, rng)
#return rval4[:, :, :, 0]
#
## TODO: this code should do it
#def random_patches(images, N, rows, cols, rng, channel_major=False):
#    if channel_major:
#        n_imgs, iF, iR, iC = images.shape
#        rval = np.empty((N, iF, rows, cols), dtype=images.dtype)
#    else:
#        n_imgs, iR, iC, iF = images.shape
#        rval = np.empty((N, rows, cols, iF), dtype=images.dtype)
#
#    srcs = rng.randint(n_imgs, size=N)
#
#    if rows > iR or cols > iC:
#        raise ValueError('cannot extract patches', (R, C))
#
#    roffsets = rng.randint(iR - rows + 1, size=N)
#    coffsets = rng.randint(iC - cols + 1, size=N)
#
#    for rv_i, src_i, ro, co in zip(rval, srcs, roffsets, coffsets):
#        if channel_major:
#            rv_i[:] = images[src_i, :, ro: ro + rows, co : co + cols]
#        else:
#            rv_i[:] = images[src_i, ro: ro + rows, co : co + cols]
#    return rval
#
