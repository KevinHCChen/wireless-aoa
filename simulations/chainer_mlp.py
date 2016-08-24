#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import computational_graph
from chainer import optimizers
from chainer import serializers

import six
import numpy as np
import math
import time
from Convert2Complex import *
from find_angle import *
plt.ion()


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units1, n_units2):
        super(MLP, self).__init__(
            l1a=L.Linear(n_in, n_units1[0]),  # first layer
            l1b=L.Linear(n_in, n_units1[0]),  # first layer
            l2a=L.Linear(n_units1[0],n_units1[1]),  # first layer
            l2b=L.Linear(n_units1[0],n_units1[1]),  # first layer
            l3=L.Linear(n_units1[1]*2, n_units2[0]),  # second layer
            l4=L.Linear(n_units2[0], n_units2[1]),  # output layer
            l5=L.Linear(n_units2[1], 2),  # output layer
        )

    def __call__(self, x, t):
        h1a = F.relu(self.l1a(x[0]))
        h1b = F.relu(self.l1b(x[1]))
        h2a = F.relu(self.l2a(h1a))
        h2b = F.relu(self.l2b(h1b))
        h_conc = F.concat((h2a,h2b))
        h3 = F.relu(self.l3(h_conc))
        h4 = F.relu(self.l4(h3))
        y = self.l5(h4)
        self.loss = F.mean_squared_error(y, t)
        #self.accuracy = F.accuracy(y, t)
        self.y = y
        return self.loss


#r = [ model(x,y) for x,y in ]

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=200, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_units = args.unit


print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')




# LOAD TRAINING DATA!
bases_set = [[((4,0), 90), ((0,-4), 0)],[((4,0), 90), ((0,4), 0)]]
num_pts = 10000
trainXs = []
testXs = []
points = np.random.uniform(-3,3,size=(num_pts,2))
points = np.array(points)

for bases in bases_set:
    X, Y = get_angles(bases,points)
    X = X/360.
    trainXs.append( X[:X.shape[0]*.8,:].astype(np.float32))
    testXs.append( X[X.shape[0]*.8:,:].astype(np.float32))

# Could be a list??? Comiter thinks so....
trainY = Y[:Y.shape[0]*.8,:].astype(np.float32)
testY = Y[Y.shape[0]*.8:,:].astype(np.float32)


model = MLP(trainXs[0].shape[1], (500,50), (200,50))

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

max_acc = 0

N = trainXs[0].shape[0]

for epoch in six.moves.range(1, n_epoch + 1):

    print('epoch', epoch)
    perm = np.random.permutation(N)

    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):

        #x = chainer.Variable(np.asarray(x_train[perm[i:i + batchsize]]))
        x = chainer.Variable(np.asarray([x[perm[i:i + batchsize]] for x in trainXs]))
        t = chainer.Variable(trainY[perm[i:i + batchsize]])

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        """
        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ))
                o.write(g.dump())
            print('graph generated')
        """

        sum_loss += float(model.loss.data) * len(t.data)
        #sum_accuracy += float(model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))


    N_test = testXs[0].shape[0]
    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):

        x = chainer.Variable(np.asarray([x[i:i + batchsize] for x in testXs]),
                             volatile='on')
        t = chainer.Variable(testY[i:i + batchsize],
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        #sum_accuracy += float(model.accuracy.data) * len(t.data)
        #max_acc = max(max_acc, sum_accuracy / N_test)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, 10./ N_test))
        #sum_loss / N_test, sum_accuracy / N_test))

x = chainer.Variable(np.asarray([x for x in testXs]),
                        volatile='on')
t = chainer.Variable(testY,
                        volatile='on')

model(x,t)
predY = model.y.data
error  = np.linalg.norm(predY - testY, axis=1)

plt.figure()#;plt.clf()
plt.subplot(2,1,1)
#plt.scatter(testY[:,0], testY[:,1], c=error)
plt.scatter(testY[:,0], testY[:,1], c=error)
plotStations(bases_set[0], 2)
plotStations(bases_set[1], 2)
plt.colorbar()
plt.ylim([-5,5])
plt.xlim([-5,5])
plt.title("Num Stations: %d" % (3))
plt.clim([0,1])
plt.show()


#plt.figure()#;plt.clf()
plt.subplot(2,1,2)
plt.scatter(predY[:,0], predY[:,1], c=error)
plotStations(bases_set[0], 2)
plotStations(bases_set[1], 2)
plt.colorbar()
plt.ylim([-5,5])
plt.xlim([-5,5])
plt.title("Num Stations: %d" % (3))
plt.clim([0,1])
plt.show()
