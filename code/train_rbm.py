#################################
#     Training the RBM          #
#################################

from __future__ import print_function
import timeit
import os
try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import theano
import theano.tensor as T

from utils import tile_raster_images

def train_rbm(rbm, x, train_set_x, batch_size, learning_rate, training_epochs, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    print('Going with ' + str(n_train_batches) + ' batches of size ' + str(batch_size))

    index = T.lscalar()    # index to a [mini]batch

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, rbm.n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm_batch is solely to update the RBM parameters
    train_rbm_batch = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm_batch'
    )

    plotting_time = 0.0
    start_time = timeit.default_timer()

    # go through training epochs
    # TEST ONLY
    training_epochs = 0
    #
    print('Starting training epochs: ' + str(training_epochs))

    for epoch in range(training_epochs):

        epoch_tic = timeit.default_timer()
         # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
             mean_cost += [train_rbm_batch(batch_index)]
        epoch_toc = timeit.default_timer()
        
        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))
        print('Time elapsed is ' + str(epoch_toc - epoch_tic))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
