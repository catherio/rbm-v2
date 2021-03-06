from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

# note: can run with
# THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python test_gpu.py

vlen = 10 * 30 * 768 # 10 * #cores * #threads/core
iters = 1000

rng = numpy.random.RandomState(22)

x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))

print(f.maker.fgraph.toposort()) # ?? what is this?

t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1-t0))
print("Result is %s" % (r,)) # what is this notation?

if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
