from __future__ import print_function

import six.moves.cPickle as pickle
from six.moves import urllib

import gzip
import os
import theano
import theano.tensor as T

from PIL import Image
import numpy as np

from rbm_obj import RBM
from load_data import load_data

# Load an existing RBM to inspect its properties

# Completely normal MNIST, I think?
#fname='/home/catherio/Classes/obda/project-code/figs/05_16_1509_mnist/rbm.pkl'

# OLD high lambda, high thresh:
# fname = '/home/catherio/Classes/obda/project-code/figs/grid-first/05_17_054338_mnist/rbm.pkl'

# OLD high lambda, low thresh:
#fname = '/home/catherio/Classes/obda/project-code/figs/grid-first/05_17_053912_mnist/rbm.pkl'

# NEW normal MNIST (15 epochs)
#fname = '/home/catherio/Classes/obda/project-code/figs/05_17_123656_mnist/rbm.pkl'

# NEW moderate lambda=0.1 (10 epochs)
#fname = '/home/catherio/Classes/obda/project-code/figs/05_17_132506_mnist/rbm.pkl'

# NEW large lambda=0.5 (10 epochs)
fname = '/home/catherio/Classes/obda/project-code/figs/05_17_143833_mnist/rbm.pkl'

# NEW overly-extreme lambda=10 (10 epochs)
#fname = '/home/catherio/Classes/obda/project-code/figs/05_17_140222_mnist/rbm.pkl'

with open(fname, 'rb') as pkl:
    rbm = pickle.load(pkl)

# Load the data it was trained on
datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[2]

# Take a look at some images
v_sample = train_set_x.get_value()[0:1000, :]
[pre_sigmoid_h1, h1_mean, h1_sample] = rbm.sample_h_given_v(v_sample)
h1_mean_val = h1_mean.eval({})
# this is (m,500) because it's all hidden units for m images

# Get the avg response of each unit, over all images here
h1_imavg = T.mean(h1_mean, 0)
h1_imavg_val = h1_imavg.eval({})
plt.figure()
plt.hist(h1_imavg_val, bins=30)
plt.title('Distribution of unit responses, averaged over images')
plt.show()

# Get the avg amount of response each image evoked, over all units
h1_havg = T.mean(h1_mean, 1)
h1_havg_val = h1_havg.eval({}) 
plt.figure()
plt.hist(h1_havg_val, bins=30)
plt.title('Distribution of evoked response to images, averaged over units')
plt.show()

# How the sparsity constraint responds to this
cost_sparsity = T.sum(abs(T.constant(rbm.sparse_thresh) - T.mean(h1_mean,0)))
cost_sparsity.eval({}) 

# Future reference: can use this code to plot two histograms on top
# of one another and still see both:
#counts,bin_edges = np.histogram(h1_mean_val, bins=20)
#bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
#plt.errorbar(bin_centers, counts, fmt='o')
#plt.show()



