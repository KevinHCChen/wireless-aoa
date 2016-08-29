import numpy as np

import matplotlib
matplotlib.use('Agg')
import utilities as util
import models as models
import data_generation as data_generation
import plotting as plotting
import json
import ast
import matplotlib.pyplot as plt
import glob
import chainer

from chainer import optimizers

use_dir = False 

if use_dir:
    # cfg_fns = "config_files/noise_model.ini"
    cfg_fns = glob.glob('expset_08292016_1pm/*')
else:
    #cfg_fns = ["config_files/noise_model.ini"]
    cfg_fns = ["config_files/broken_structured.ini"]


for cfg_fn in cfg_fns:

    config, dir_name = util.load_configuration(cfg_fn)

    params = util.create_param_dict(config)

    print params

    if params['exp_details__interactive']:
        plt.ion()



    # generate mobile points, base stations, and angles
    mobiles, bases, angles = data_generation.generate_data(params['data__num_pts'],
                                                           params['data__num_stations'],
                                                           params ['data__ndims'],
                                                           pts_r=3.9, bs_r=4,
                                                           bs_type=params['data__bs_type'])

    #angles = data_generation.add_noise(angles, col_idxs=range(angles.shape[1]), noise_params={'mean': 0, 'std': 1} )
    # split data

    trainXs, trainY, testXs, testY = util.test_train_split(angles, mobiles)

    #assume trainXs num pts by 4


    models_l = []
    models_l.append(models.BaseMLP(trainXs[0].shape[1]/2, params['NN__network_size'][0],
                           params['data__ndims']) )

    models_l.append(models.BaseMLP(trainXs[0].shape[1]/2, params['NN__network_size'][0],
                           params['data__ndims']))

    output_testXs = []
    output_trainXs= []
    for i, model in enumerate(models_l):
        # Setup optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model)


        # train model
        #model, loss = models.train_model(model, list(trainXs[0][:,(i*2):(i*2)+2]),
        #                                trainY, list(testXs[0][:,(i*2):(i*2)+2]), testY,
        model, loss = models.train_model(model, trainXs[0][:,(i*2):(i*2)+2],
                                         trainY, testXs[0][:,(i*2):(i*2)+2], testY,
                                n_epoch=params['NN__n_epochs'],
                                batchsize=params['NN__batchsize'],
                                max_flag=params['NN__take_max'])

        x = chainer.Variable(np.asarray(trainXs[0][:,(i*2):(i*2)+2]))
        output_trainXs.append( model.forward(x) )

        x = chainer.Variable(np.asarray(testXs[0][:,(i*2):(i*2)+2]))
        output_testXs.append( model.forward(x) )


    trainXs = np.hstack([x.data for x in output_trainXs])
    testXs = np.hstack([x.data for x in output_testXs])
    print trainXs.shape, testXs.shape
    print 'got to 1'
    model = models.BaseMLP(trainXs.shape[1], params['NN__network_size'][1],
                           params['data__ndims'])

    print trainXs.shape
    print trainY.shape
    print 'got to 2'

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    model, loss = models.train_model(model, trainXs,
                                         trainY, testXs, testY,
                                n_epoch=params['NN__n_epochs'],
                                batchsize=params['NN__batchsize'],
                                max_flag=params['NN__take_max'])


    f = open(dir_name + 'loss.txt', 'w')
    f.write("%f" % (loss))
    f.close()


    """
    # generate mobile points, base stations, and angles
    mobiles, bases, angles = data_generation.generate_data(10000,
                                                           params['data__num_stations'],
                                                           params ['data__ndims'],
                                                           pts_r=3.9, bs_r=4,
                                                           bs_type=params['data__bs_type'])

    trainXs, trainY, testXs, testY = util.test_train_split(angles, mobiles, 0.)
    """

    # test model

    predY, error = models.test_model(model, testXs, testY)

    plotting.plot_error(testY, predY, error, bases,
                        "Num Stations: %d" % (params['data__num_stations']),
                        params['exp_details__save'], dir_name)


    #error = np.exp(error)
    #plotting.plot_error(testY, predY, error, bases, "Num Stations: %d" % (params['data__num_stations']))

    # TODO: write results file to directory
    # if params['exp_details__save']:
    #     print "****** NEED TO IMPLEMENT SAVING ********"
    # else:
    #     print "****** Not saving!!!! ****** "

    # print out warning if figures not saved
    if params['exp_details__save']:
        print "****** Figures saved to directory %s ********" % (dir_name)
    else:
        print "****** Not saving!!!! ****** "
        print "If you would like to save, change the config file"





