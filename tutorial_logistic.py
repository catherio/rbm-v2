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
b = theano.shared(0.0,'b') # watch out for 0 (which is an int64) versus 0.
w = theano.shared(rng.randn(nfeats),'w') # one bias, many weights

p = 1 / (1 + T.exp(-(b + T.dot(x,w)))) # predicted label
    # Matrix multiplication is "dot" here
    # Soft skills notes here:
    #     build and compile functions for pieces of the overall expression
    #     use .get_value() to use shared variables when computing
    # 

# Define loss function
loss = -y * T.log(p) - (1-y) * T.log(1-p) # NLL

# Cost is like loss, but also includes regularization,
# and sums over all samples
cost = loss.mean() + (0.01 * w.get_value()**2).sum()

# Gradients are going to need to come into play, and wow, I do not have to compute them!
gw, gb = T.grad(cost, [w,b])

# Training
learningrate = 0.1
trainstep = theano.function(inputs=[x,y],
                            outputs=[p, cost],
                            updates=[[w, w-0.1*gw], [b, b-0.1*gb]]) 
nsteps = 500
dispevery = 50
for ii in range(nsteps):
    [thisp, thiscost] = trainstep(D[0], D[1])
    if ii % dispevery == 0:
        print 'Step ', ii, ', cost ', thiscost

# How'd it do?

prediction = p > 0.5
   # seems like even simple additions like this need two steps,
   # one to set up a variable and the other to make a function of it
predict = theano.function([x], prediction)

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))



    

