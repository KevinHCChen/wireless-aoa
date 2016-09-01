import numpy as np
import json
import ast
import glob
import argparse
import noise_models as noise_models 
from chainer import optimizers


parser = argparse.ArgumentParser(description='Driver for 5G Experiments')
parser.add_argument('--showfig', '-g', dest='showfig', action='store_true',
                    help='Show the figure')
parser.add_argument('--configfile', '-c', dest='configfile', type=str,
                    help='Which config file to use')
args = parser.parse_args()

showfig = args.showfig
configfile = args.configfile
if not showfig:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

import utilities as util
import models as models
import data_generation as data_generation
import plotting as plotting


use_dir = False

if configfile:
    cfg_fns = [configfile]
elif use_dir:
    # cfg_fns = "config_files/noise_model.ini"
    cfg_fns = glob.glob('expset_09012016_10am/*')
else:
    #cfg_fns = ["config_files/noise_baseModel.ini"]
    #cfg_fns = ["config_files/broken_structured.ini"]
    cfg_fns = ["config_files/baseModel2D.ini"]
    # cfg_fns = ["expset_08312016_10pm/exp1_setting3.ini"]*10 + ["expset_08312016_10pm/exp1_setting10.ini"]*10


for cfg_fn in cfg_fns:
    print "CFG: ", cfg_fn
    config, dir_name = util.load_configuration(cfg_fn)

    params = util.create_param_dict(config)

    print params

    if params['exp_details__interactive']:
        plt.ion()


    all_predY = None
    all_error = None
    mean_errors = []
    for iter_number in range(params['exp_details__num_iterations_per_setting']):
        # generate mobile points, base stations, and angles
        mobiles, bases, angles = data_generation.generate_data(params['data__num_pts'],
                                                               params['data__num_stations'],
                                                               params ['data__ndims'],
                                                               pts_r=3.9, bs_r=4,
                                                               bs_type=params['data__bs_type'], points_type="random")


        # IMPORTANT: remember to add noise before replicating data (e.g., for snbp-mlp)
        if params['noise__addnoise_train']:
            angles = noise_models.add_noise_dispatcher(angles, params['noise__noise_model'], params['data__ndims'], base_idxs=params['noise__bases_to_noise'], 
                                                            noise_params=params['noise__noise_params'])

        if params['NN__type'] == 'snbp-mlp':
            rep_idxs = [[0,2],[1,2]]
            angles = data_generation.replicate_data(angles, params['data__ndims'],  rep_idxs)


        # split data
        trainXs, trainY, testXs, testY = util.test_train_split(angles, mobiles)


        if params['NN__type'] == "bmlp":
            model = models.BaseMLP(np.hstack(trainXs).shape[1], params['NN__network_size'],
                                   params['data__ndims'])
        elif params['NN__type'] == 'smlp':
            model = models.StructuredMLP(None, params['NN__network_size'][0],
                                         params['NN__network_size'][1], params['data__ndims'],
                                         [[0,1],[2,3]])
        elif params['NN__type'] == 'snbp-mlp':
            model = models.NBPStructuredMLP(trainXs[0].shape[1], params['NN__network_size'][0],
                                            params['NN__network_size'][1], params['data__ndims'],
                                            len(rep_idxs))

        # train model
        #model, loss = models.train_model(model, trainXs, trainY, testXs, testY,
        loss = model.trainModel(trainXs, trainY, testXs, testY,
                                   n_epoch=params['NN__n_epochs'],
                                   batchsize=params['NN__batchsize'],
                                   max_flag=params['NN__take_max'])

        f = open(dir_name + 'loss_iteration%d.txt' % (iter_number), 'w')
        f.write("%f" % (loss))
        f.close()


        # generate mobile points, base stations, and angles
        mobiles, bases, angles = data_generation.generate_data(50*50,
                                                               params['data__num_stations'],
                                                               params ['data__ndims'],
                                                               pts_r=3, bs_r=4,
                                                               bs_type=params['data__bs_type'], points_type="grid")


        if params['noise__addnoise_test']:
            angles = noise_models.add_noise_dispatcher(angles, params['noise__noise_model'], params['data__ndims'], base_idxs=params['noise__bases_to_noise'], 
                                                            noise_params=params['noise__noise_params'])

        if params['NN__type'] == 'snbp-mlp':
            angles = data_generation.replicate_data(angles, params['data__ndims'],  rep_idxs)

        trainXs, trainY, testXs, testY = util.test_train_split(angles, mobiles, 0.)

        # test model
        #predY, error = models.test_model(model, testXs, testY)
        predY, error = model.testModel(testXs, testY)

        f = open(dir_name + 'error_iteration%d.txt' % iter_number, 'w')
        f.write("Mean Error: %f\n" % (np.mean(error)))
        f.write("Error Standard Deviation: %f\n" % (np.std(error)))
        f.close()

        mean_errors.append(np.mean(error))



        plotting.plot_error(testY, predY, error, bases,
                            "Num Stations: %d" % (params['data__num_stations']),
                            params['exp_details__save'], dir_name, iter_number)

        if all_predY == None:
            all_predY = np.zeros((predY.shape[0], predY.shape[1], params['exp_details__num_iterations_per_setting']))
        if all_error == None:
            all_error = np.zeros((error.shape[0], params['exp_details__num_iterations_per_setting']))   

        all_predY[:,:,iter_number] = predY
        all_error[:,iter_number] = error

    f = open(dir_name + 'error_average.txt', 'w')
    f.write("Mean Error: %f\n" % (np.mean(mean_errors)))
    f.close()


    f = open('resultsdata.npz', 'w')
    np.savez(f, all_predY=all_predY, all_error=all_error)
    f.close()


    # print out warning if figures not saved
    if params['exp_details__save']:
        print "****** Figures saved to directory %s ********" % (dir_name)
    else:
        print "****** Not saving!!!! ****** "
        print "If you would like to save, change the config file"





