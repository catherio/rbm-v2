
#################################
#     Sampling from the RBM     #
#################################

from __future__ import print_function
import os
import datetime
import math

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import theano
import theano.tensor as T
from utils import tile_raster_images

def sample_rbm(rbm, test_set_x, n_chains, n_samples, output_folder, rng):

    # set output directory
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)

    img_side = math.sqrt(rbm.n_visible)
    tile_space = 1

    image_data = numpy.zeros(
        ((img_side + tile_space) * n_samples + 1,
         (img_side + tile_space) * n_chains - 1),
        dtype='uint8'
    )

    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[(img_side + tile_space) * idx :
                    (img_side + tile_space) * idx + img_side,
                    :] = tile_raster_images(
            X=vis_mf,
            img_shape=(img_side, img_side),
            tile_shape=(1, n_chains),
            tile_spacing=(tile_space, tile_space)
        )

    # construct image
    image = Image.fromarray(image_data)
    
    nowstr = datetime.datetime.now().strftime("%m_%d_%I:%M%p")
    image.save('samples_' + nowstr + '.png')

