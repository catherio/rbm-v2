from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os

import numpy as np

# where is it located
dataloc = '/home/cao324/rbm/data/vanhateren'

# load it; let's pretend this takes a very long time
fname = 'imk00001.iml'
s = open(os.path.join(dataloc, fname), 'rb').read()

# manipulate the loaded image
img = np.fromstring(s, dtype='uint16').byteswap()
img = img.reshape([1024, 1536])

# look at it
# I assume this involves using PIL
# and saving out a .png or similar
