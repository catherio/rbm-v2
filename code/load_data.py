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

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join("..", "data", dataset)
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
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

################ VAN HATEREN #####################

def load_vanhateren(dataset):
    data_dir, data_file = os.path.split(dataset)
    assert(data_file == 'vanhateren')

    data_dir = os.path.join('..', 'data', 'vanhateren')
        # alternate data_dir not currently working
    imc_or_iml = 'iml'
    nIms = 500 # paper had 2000

    # Get the data, downloading if needed, but don't force download
    ims = get_vanhateren(nIms, imc_or_iml, data_dir, False)
    
    train_size = 1000 # MNIST was 50,000, paper had 100,000
    test_size = 1000
    # We don't actually need any validation set for this purpose

    # Split into patches
    patchsz = 14 # not sure where I got 32 from earlier

    train_x = make_patches(ims, patchsz, train_size)
    train_x = np.reshape(train_x,
                        [train_x.shape[0], patchsz*patchsz])

    test_x = make_patches(ims, patchsz, test_size)
    test_x = np.reshape(test_x,
                        [test_x.shape[0], patchsz*patchsz])

    # Note on scaling the intensity values:
    # These are now scaled in (0.0, 1.0)
    # but you can multiply by 255 to view with im.show

    return [[train_x, []],
            [[], []],
            [test_x, []]]


def get_vanhateren(nIms, imc_or_iml, data_dir, force):
    filename_spec = 'imk%.5d.' + imc_or_iml

    which_ims = range(1,nIms+1)
    max_file = os.path.join(data_dir, filename_spec % which_ims[-1])

    if (not os.path.isfile(max_file) or force):
        origin = (
        'http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/' + imc_or_iml + '/'
        ) 
        
        print('Downloading data from %s' % origin)
        for i in which_ims:
            getfile = os.path.join(origin, filename_spec % i)
            putfile = os.path.join(data_dir, filename_spec % i)
            urllib.request.urlretrieve(getfile, putfile)

        print('Done downloading')

    datasz = [1024, 1536]
    datatype = 'uint16'
    datamax = 4095.0 # the max to clip at and divide by

    def read_image(dataloc):
        # datasz, datatype, datamax are inherited from context
        readfile = open(dataloc, 'rb').read()
        im_array = np.fromstring(readfile, dtype=datatype).byteswap()
        im_array = im_array.reshape(datasz)
        im_array = im_array.astype('float32')
        im_array = im_array / datamax
        im_array[im_array > 1] = 1 # clip outliers
        return im_array
        # NOTE: this format can be used with
        # im = Image.fromarray(im_array) if desired
        # and inspected with im.show() or im.save('tmp.gif')


    all_files = [os.path.join(data_dir, filename_spec % i) for i in which_ims]
    ims = map(read_image, all_files)
    return ims


def make_patches(ims, patchsz, n_patches):
    [imrows, imcols] = ims[0].shape
    assert(patchsz < imrows)
    assert(patchsz < imcols)

    patches = np.empty([n_patches, patchsz, patchsz], dtype=ims[0].dtype)

    rng = np.random
    src_ims = rng.randint(0, len(ims), n_patches)
    row_offsets = rng.randint(imrows-patchsz+1, size=n_patches)
    col_offsets = rng.randint(imcols-patchsz+1, size=n_patches)

    for put_here, src, r, c in zip(patches, src_ims, row_offsets, col_offsets):
        put_here[:] = ims[src][r:r+patchsz, c:c+patchsz]

    return patches


################ MNIST #####################

# train_set is [(50000, 784), (50000,)] (where 784 = 28*28)
# valid_set is [(10000, 784), (10000,)]
# test_set is [(10000, 784), (10000,)]

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
        [train_set, valid_set, test_set] = pickle.load(f)
        
    # TODO FIXME: TEMPORARY TIME-SAVER for super speedy training
    train_set = (train_set[0][0:10000,:], train_set[1][0:10000,])
    valid_set = (valid_set[0][0:10000,:], valid_set[1][0:10000,])
    test_set =  (test_set[0][0:10000,:],  test_set[1][0:10000,])

    return [train_set, valid_set, test_set]



