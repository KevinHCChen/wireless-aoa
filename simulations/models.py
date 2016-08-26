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
#plt.ion()



# Network definition
class StructuredMLP(chainer.Chain):

    def __init__(self, n_in, n_units1, n_units2):
        
        super(StructuredMLP, self).__init__(
            l1a=L.Linear(n_in, n_units1[0]),  # first layer
            l1b=L.Linear(n_in, n_units1[0]),  # first layer
            l2a=L.Linear(n_units1[0],n_units1[1]),  # first layer
            l2b=L.Linear(n_units1[0],n_units1[1]),  # first layer
            l3=L.Linear(n_units1[1]*2, n_units2[0]),  # second layer
            l4=L.Linear(n_units2[0], n_units2[1]),  # output layer
            l5=L.Linear(n_units2[1], 2),  # output layer
        )
        self.name = 'smlp'

    def __call__(self, x, t):
        x = x[0]
        # print x[:,0:2].shape
        # print x[:,2:4].shape
        # assert False
        h1a = F.relu(self.l1a(x[:,0:2]))
        h1b = F.relu(self.l1b(x[:,2:4]))
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


        # h1a = F.relu(self.l1a(x[0]))
        # h1b = F.relu(self.l1b(x[1]))
        # h2a = F.relu(self.l2a(h1a))
        # h2b = F.relu(self.l2b(h1b))
        # h_conc = F.concat((h2a,h2b))
        # h3 = F.relu(self.l3(h_conc))
        # h4 = F.relu(self.l4(h3))
        # y = self.l5(h4)
        # self.loss = F.mean_squared_error(y, t)
        # #self.accuracy = F.accuracy(y, t)
        # self.y = y
        # return self.loss


# Network definition
class BaseMLP(chainer.ChainList):

    def __init__(self, n_in, n_units, n_out):
        
        super(BaseMLP, self).__init__()
        self.add_link(L.Linear(n_in, n_units[0]))
        for i in range(len(n_units)-1):
            # layers_l.append(L.Linear(n_units[i], n_units[i+1]))
            self.add_link(L.Linear(n_units[i], n_units[i+1]))
        self.add_link(L.Linear(n_units[-1], n_out))
        self.num_layers = len(n_units) + 2
        self.name = 'bmlp'

    def __call__(self, x, t):
        h_i = x
        for i in range(self.num_layers-2):
            h_i = F.relu(self[i](h_i))

        y = self[-1](h_i)

        self.loss = F.mean_squared_error(y, t)
        self.y = y

        return self.loss



def train_model(model, trainXs, trainY, testXs, testY, n_epoch=200, batchsize=100):
    print "TrainX: ", trainXs[0].shape
    print "TrainY: ", trainY.shape
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
            if model.name == 'smlp':
                x = chainer.Variable(np.asarray([x[perm[i:i + batchsize]] for x in trainXs]))
            elif model.name == 'bmlp':
                x = chainer.Variable(np.asarray(np.hstack(trainXs)[perm[i:i + batchsize]]))
            else:
                assert False, "Error in models.py train_model(): Not a valid model"

            t = chainer.Variable(trainY[perm[i:i + batchsize]])

            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(model, x, t)

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

            if model.name == 'smlp':
                x = chainer.Variable(np.asarray([x[i:i + batchsize] for x in testXs]),
                                 volatile='on')
            elif model.name == 'bmlp':
                x = chainer.Variable(np.asarray(np.hstack(testXs)[i:i + batchsize]),
                                 volatile='on')
            else:
                assert False, "Error in models.py train_model(): Not a valid model"

            t = chainer.Variable(testY[i:i + batchsize],
                                 volatile='on')
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            #sum_accuracy += float(model.accuracy.data) * len(t.data)
            #max_acc = max(max_acc, sum_accuracy / N_test)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, 10./ N_test))
            #sum_loss / N_test, sum_accuracy / N_test))

    return model


def test_model(model, testXs, testY):
    if model.name == 'smlp':
        x = chainer.Variable(np.asarray([x for x in testXs]),
                            volatile='on')
    elif model.name == 'bmlp':
        x = chainer.Variable(np.asarray(np.hstack(testXs)),
                            volatile='on')
    else:
        assert False, "Error in models.py test_model(): Not a valid model"
    t = chainer.Variable(testY,
                            volatile='on')

    print(x.shape)
    print(t.shape)
    model(x,t)
    predY = model.y.data
    error  = np.linalg.norm(predY - testY, axis=1)
    return predY, error




