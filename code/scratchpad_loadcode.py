
from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
from PIL import Image

import numpy as np

# NOTE TO SELF:
# im = Image.fromarray(im_array) if desired
# can be inspected with im.show() or im.save('tmp.gif')

# NOTE FROM SKDATA: they use dictionaries of properties to represent
# each image, which contains the location, the datatype...
# ALSO, the overall collection is object-oriented, and defines the ability
# to load individual items if they're missing. That's really cool
# and something I could do in the future but GOSH not now!!

# TODO ok yo, this is the scratchpad of where make_patches goes!

# START with the images

data_dir = '../data/vanhateren/'
# alternate data_dir not currently working
imc_or_iml = 'iml'
nIms = 20 # TODO: specified in the paper as 2000

# Get the data, downloading if needed, but don't force download
ims = get_vanhateren(nIms, imc_or_iml, data_dir, False)

# TODO: Load the patches
patchsz = 32
n_patches = 30 # TODO: specified in the paper as 10,000
#patches = make_patches(ims, patchsz, n_patches)



