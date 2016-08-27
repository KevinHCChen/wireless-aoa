import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import computational_graph
from chainer import optimizers
from chainer import serializers
import copy

import six
import numpy as np
import math
import time
from Convert2Complex import *
#plt.ion()


# Network definition
class StructuredMLP(chainer.ChainList):

    def __init__(self, n_in, n_lower, n_upper, n_out, col_idxs_l):

        super(StructuredMLP, self).__init__()

        for col_idxs in col_idxs_l:
            self.add_link(L.Linear(len(col_idxs), n_lower[0]))
            for i in range(len(n_lower)-1):
                self.add_link(L.Linear(n_lower[i], n_lower[i+1]))
        self.add_link(L.Linear(n_lower[-1]*len(col_idxs_l), n_upper[0]))

        for i in range(len(n_upper)-1):
            self.add_link(L.Linear(n_upper[i], n_upper[i+1]))
        self.add_link(L.Linear(n_upper[-1], n_out))
        self.num_layers_lower = len(n_lower)
        self.num_layers_upper = len(n_upper)

        self.col_idxs_l = col_idxs_l

        self.name = 'smlp'

    def __call__(self, x, t):

        x = x[0]
        h_i = x

        cnt = 0
        lower_levels = []
        for col_idxs in self.col_idxs_l:
            h_i = F.relu(self[cnt](x[:,col_idxs]))
            cnt +=1
            for i in range(1,self.num_layers_lower):
                h_i = F.relu(self[cnt](h_i))
                cnt += 1
            lower_levels.append(copy.deepcopy(h_i))

        h_i = F.concat(tuple(lower_levels))
        for i in range(self.num_layers_upper):
            h_i = F.relu(self[cnt](h_i))
            cnt += 1


        y = self[-1](h_i)
        self.loss = F.mean_squared_error(y, t)
        self.y = y
        return self.loss



# Network definition
class BaseMLP(chainer.ChainList):

    def __init__(self, n_in, n_units, n_out):

        super(BaseMLP, self).__init__()
        self.add_link(L.Linear(n_in, n_units[0]))
        for i in range(len(n_units)-1):
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




