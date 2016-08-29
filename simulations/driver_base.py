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

from chainer import optimizers

use_dir = True

if use_dir:
    # cfg_fns = "config_files/noise_model.ini"
    cfg_fns = glob.glob('expset_08292016_1pm/*')
else:
    cfg_fns = ["config_files/noise_model.ini"]


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


    if params['NN__type'] == "bmlp":
        model = models.BaseMLP(np.hstack(trainXs).shape[1], params['NN__network_size'],
                               params['data__ndims'])
    elif params['NN__type'] == 'smlp':
        model = models.StructuredMLP(None, params['NN__network_size'][0],params['NN__network_size'][1], params['data__ndims'],
                                 [[0,1],[2,3]])

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    # train model
    model = models.train_model(model, trainXs, trainY, testXs, testY,
                               n_epoch=params['NN__n_epochs'],
                               batchsize=params['NN__batchsize'],
                               max_flag=params['NN__take_max'])


    # generate mobile points, base stations, and angles
    mobiles, bases, angles = data_generation.generate_data(10000,
                                                           params['data__num_stations'],
                                                           params ['data__ndims'],
                                                           pts_r=3.9, bs_r=4,
                                                           bs_type=params['data__bs_type'])

    trainXs, trainY, testXs, testY = util.test_train_split(angles, mobiles, 0.)

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





