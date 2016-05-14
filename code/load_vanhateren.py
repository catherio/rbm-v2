
from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
from PIL import Image

import numpy as np

# NOTE TO SELF: this can be inspected with im.show() or im.save('tmp.gif')

# NOTE FROM SKDATA: they use dictionaries of properties to represent
# each image, which contains the location, the datatype...
# ALSO, the overall collection is object-oriented, and defines the ability
# to load individual items if they're missing. That's really cool
# and something I could do in the future but GOSH not now!!

# TODO ok yo, this is the scratchpad of where make_patches goes!

# Choose source images, randomly

# TODO: this code should do it
def random_patches(images, N, rows, cols, rng, channel_major=False):
    if channel_major:
        n_imgs, iF, iR, iC = images.shape
        rval = np.empty((N, iF, rows, cols), dtype=images.dtype)
    else:
        n_imgs, iR, iC, iF = images.shape
        rval = np.empty((N, rows, cols, iF), dtype=images.dtype)

    srcs = rng.randint(n_imgs, size=N)

    if rows > iR or cols > iC:
        raise ValueError('cannot extract patches', (R, C))

    roffsets = rng.randint(iR - rows + 1, size=N)
    coffsets = rng.randint(iC - cols + 1, size=N)

    for rv_i, src_i, ro, co in zip(rval, srcs, roffsets, coffsets):
        if channel_major:
            rv_i[:] = images[src_i, :, ro: ro + rows, co : co + cols]
        else:
            rv_i[:] = images[src_i, ro: ro + rows, co : co + cols]
    return rval
