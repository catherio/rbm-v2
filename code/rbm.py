"""Skeleton code borrowed from the Theano tutorial on RBM's.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""

# Import all the things!
from __future__ import print_function
import timeit
import datetime
import os
import inspect

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# Modular components
from rbm_obj import RBM
from load_data import load_data
from sample_rbm import sample_rbm
from train_rbm import train_rbm

# Continued below
def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=4, output_folder='../figs',
             n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano.
    This is demonstrated on MNIST.
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path to the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    """

    #################################
    #      Set output folder        #
    #################################

    nowstr = datetime.datetime.now().strftime("%m_%d_%H%M")
    data_dir, data_file = os.path.split(dataset)
    data_name = data_file.split('.')[0]

    output_folder = os.path.join(output_folder, nowstr + '_' + data_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_folder = os.path.realpath(output_folder)

    cur = os.getcwd()
    os.chdir(output_folder)
    f = inspect.currentframe()    
    with open("params.txt", "w") as txt:
        txt.write(str(inspect.getargvalues(f).locals))
    os.chdir(cur)


    #################################
    #      Downloading data         #
    #################################
    print('Start dataset load')
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # At this point, train_set_x is a theano Tensor
    input_size = train_set_x.get_value().shape[1]

    print('End dataset load')

    #################################
    #   Constructing the RBM        # 
    #################################
    print('Constructing the RBM')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # allocate symbolic variable for the data
    x = T.matrix('x')  # the data is presented as rasterized images

    # construct the RBM class
    rbm = RBM(input=x, n_visible=input_size,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    #################################
    #     Training the RBM          #
    #################################

    print('Training:')
    train_rbm(rbm, x, train_set_x, batch_size, learning_rate, training_epochs, output_folder)

    #################################
    #     Sampling from the RBM     #
    #################################
    print('Sampling:')
    sample_rbm(rbm, test_set_x, n_chains, n_samples, output_folder, rng)

#if __name__ == '__main__':
#    test_rbm()

# Can set training_epochs = 0 to skip training
# 
