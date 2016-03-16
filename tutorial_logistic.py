import numpy
import theano
import theano.tensor as T 

# Give the random number generator a name
rng = numpy.random # this is just a package

# Set parameters
ndata = 10 #400
nfeats = 4 #784 # why 784?

# Generate dataset, with categories
# TODO don't know why they made this a tuple of matrices, but ok
D = (rng.randn(ndata, nfeats),
     rng.randint(size=ndata, low=0, high=2))

# Define and initialize symbolic variables to define model
x = T.matrix('x') # input values
y = T.vector('y') # true label
# b = T.dscalar('b')
# w = T.dscalar('w') # aha, nope, these need to get trained!!
b = theano.shared(0,'b') 
w = theano.shared(rng.randn(nfeats),'w') # one bias, many weights

p = 1 / (1 + T.exp(-(b + T.dot(x,w)))) # predicted label
    # Matrix multiplication is "dot" here

# Define loss function
loss = -y * T.log(p) - (1-y) * T.log(1-p) # NLL

# Gradients are going to need to come into play, and wow, I do not have to compute them!

grd = T.grad(cost=TODO, wrt='w')

# Compile
lgstc = theano.function([x], p)

# Training


# Testing
