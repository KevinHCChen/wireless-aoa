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

    def trainModel(self, trainXs, trainY, testXs, testY, n_epoch=200, batchsize=100, max_flag=False):
        self.model, loss = train_model(trainXs, trainY, testXs, testY, n_epoch=n_epoch,
                                       batchsize=batchsize, max_flag=max_flag)


class NBPStructuredMLP():

    def __init__(self, n_in, n_lower, n_upper, n_out):

        num_pairs = 2
        ndim = n_out
        self.lower_models_l = []
        self.lower_models_l.append(BaseMLP(n_in/num_pairs, n_lower, ndim))
        self.lower_models_l.append(BaseMLP(n_in/num_pairs, n_lower, ndim))

        self.upper_model = BaseMLP(n_lower[-1]*num_pairs, n_upper, ndim)

    def trainModel(self, trainXs, trainY, testXs, testY, n_epoch=200, batchsize=100, max_flag=False):

        output_testXs = []
        output_trainXs= []
        for i, model in enumerate(self.lower_models_l):
            # Setup optimizer
            optimizer = optimizers.Adam()
            optimizer.setup(model)

            tmp_trainXs = []
            tmp_testXs = []
            tmp_trainXs.append(trainXs[0][:,(i*ndim):(i*ndim)+ndim])
            tmp_testXs.append(testXs[0][:,(i*ndim):(i*ndim)+ndim])

            model, loss = train_model(model, tmp_trainXs, trainY,
                                            tmp_testXs, testY,
                                          n_epoch=n_epoch,
                                          batchsize=batchsize,
                                          max_flag=max_flag)

            x = chainer.Variable(np.asarray(tmp_trainXs[0]))
            output_trainXs.append( model.forward(x) )

            x = chainer.Variable(np.asarray(tmp_testXs[0]))
            output_testXs.append( model.forward(x) )


        trainXs = np.hstack([x.data for x in output_trainXs])
        testXs = np.hstack([x.data for x in output_testXs])

        tmp_trainXs = []
        tmp_testXs = []
        tmp_trainXs.append(trainXs)
        tmp_testXs.append(testXs)
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        self.upper_model, loss = train_model(self.upper_model, tmp_trainXs, trainY,
                                          tmp_testXs, testY,
                                          n_epoch=n_epoch,
                                          batchsize=batchsize,
                                          max_flag=max_flag)

        return loss

    def forward(self, X):

        output_testXs = []
        for i, model in enumerate(self.lower_models_l):
            tmp_testXs = []
            tmp_testXs.append(X[0][:,(i*ndim):(i*ndim)+ndim])

            x = chainer.Variable(np.asarray(tmp_testXs[0]))
            output_testXs.append( model.forward(x) )



        testXs = np.hstack([x.data for x in output_testXs])

        tmp_testXs = []
        tmp_testXs.append(testXs)
        return tmp_testXs


    def testModel(self, X, Y):
        res = self.forward(X)
        #predY, error = test_model(self.upper_model, X, Y)
        predY, error = test_model(self.upper_model, res, Y)
        return predY, error




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

    def forward(self,x):
        h_i = x
        for i in range(self.num_layers-2):
            h_i = F.relu(self[i](h_i))

        return h_i

    def trainModel(self, trainXs, trainY, testXs, testY, n_epoch=200, batchsize=100, max_flag=False):

        self.model, loss = train_model(self, trainXs, trainY, testXs, testY,
                                       n_epoch, batchsize, max_flag=max_flag)
        return loss


    def testModel(self, X, Y):
        #res = self.forward(X)
        predY, error = test_model(self, X, Y)
        #predY, error = test_model(self.upper_model, res, Y)
        return predY, error


def train_model(model, trainXs, trainY, testXs, testY, n_epoch=200, batchsize=100, max_flag=False):
    print "TrainX: ", trainXs[0].shape
    print "TrainY: ", trainY.shape
    print "Test:", testXs[0].shape

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    max_acc = 0

    N = trainXs[0].shape[0]

    best_model = None
    min_metric = float('inf')

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
                #x = chainer.Variable(np.asarray(trainXs[perm[i:i + batchsize]]))
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
                #x = chainer.Variable(np.asarray(testXs[i:i + batchsize]),
            else:
                assert False, "Error in models.py train_model(): Not a valid model"

            t = chainer.Variable(testY[i:i + batchsize],
                                 volatile='on')
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            #sum_accuracy += float(model.accuracy.data) * len(t.data)
            #max_acc = max(max_acc, sum_accuracy / N_test)

        if (sum_loss / N_test) < min_metric:
            best_model = copy.deepcopy(model)
            min_metric = (sum_loss / N_test)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, 10./ N_test))
            #sum_loss / N_test, sum_accuracy / N_test))

    if max_flag:
        return best_model, min_metric
    else:
        return model, sum_loss / N_test


def test_model(model, testXs, testY):
    if model.name == 'smlp':
        x = chainer.Variable(np.asarray([x for x in testXs]),
                            volatile='on')
    elif model.name == 'bmlp':
        x = chainer.Variable(np.asarray(np.hstack(testXs)),
                            volatile='on')
        #x = chainer.Variable(np.asarray(testXs),
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




