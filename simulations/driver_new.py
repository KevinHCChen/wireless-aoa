import numpy as np
import json
import ast
import glob
import argparse
import itertools
import phase_noise_models as noise_models
import datetime
from chainer import optimizers
import os
import pandas as pd


parser = argparse.ArgumentParser(description='Driver for 5G Experiments')
parser.add_argument('--showfig', '-g', dest='showfig', action='store_true',
                    help='Show the figure')
parser.add_argument('--configfile', '-c', dest='configfile', type=str,
                    help='Which config file to use')
parser.add_argument('--configfile_dir', '-d', dest='configfile_dir', type=str,
                    help='Which directory of config files to use')
parser.add_argument('--startidx', '-s', dest='startidx', type=int,
                    help='Which file to start at')
parser.add_argument('--endidx', '-e', dest='endidx', type=int,
                    help='Which file to end at')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', 
                    help='Print out verbose')
args = parser.parse_args()

showfig = args.showfig
configfile = args.configfile
startidx = args.startidx
endidx = args.endidx
configfile_dir = args.configfile_dir
startidx = args.startidx
endidx = args.endidx
verbose = args.verbose

import utilities as util
import models as models
import data_generation as data_generation
import plotting as plotting

num_antennas_per_bs = 4
used_antennas_per_bs = 4
use_dir = False

def normalize(d, d_min, d_max):
    return  (d + d_min)/(d_max+d_min)

def normalize_cols(d):
    for i in range(d.shape[1]):
        d[:,i] = (d[:, i] + np.abs(np.min(d[:,i])))/(np.max(d[:,i]) + np.abs(np.min(d[:,i])))
    return d

def normalize_cols_data(d, d_min, d_max):
    for i in range(d.shape[1]):
        d[:,i] = (d[:, i] + d_min[i])/(d_max[i] + d_min[i])
    return d


def normalize_unit(d):
    for i in range(d.shape[1]):
        d[:,i] = (d[:, i])/ np.linalg.norm(d[:,i]) 
    return d

def normalize_unitAndCol(d, d_min=None, d_max=None):
    d = normalize_unit(d)
    if d_min is None:
        d_min = np.abs(np.min(d, axis=0))
        d_max= np.max(d, axis=0)
    d = normalize_cols_data(d, d_min, d_max)
    return d, d_min, d_max

def zero_mean(d):
    for i in range(d.shape[1]):
        d[:,i] = d[:,i] - np.mean(d[:,i])
    return d 


if configfile:
    cfg_fns = [configfile]
elif configfile_dir:
    if configfile_dir[-1] != '/':
        configfile_dir += '/'
    cfg_fns = glob.glob(configfile_dir + '*')
    cfg_fns.sort()
    if endidx:
        assert startidx <= endidx, "Startidx is greater than endidx...not judging, just letting you know..."
        cfg_fns = cfg_fns[startidx:endidx+1]
elif use_dir:
    # cfg_fns = "config_files/noise_model.ini"
    cfg_fns = glob.glob('exp_bm_gaussian_11am/*')
    cfg_fns.sort()
    if endidx:
        assert startidx <= endidx, "Startidx is greater than endidx...not judging, just letting you know..."
        cfg_fns = cfg_fns[startidx:endidx]
    #cfg_fns = glob.glob('test_batch/*')
else:
    #cfg_fns = ["config_files/noise_baseModel.ini"]
    #cfg_fns = ["config_files/broken_structured.ini"]
    cfg_fns = ["config_files/baseModel2D.ini"]
    #cfg_fns = ["expset_08312016_10pm/exp1_setting3.ini"]*1 + ["expset_08312016_10pm/exp1_setting10.ini"]*1

df_all = pd.DataFrame()

type_of_xs = 'phases'


for cfg_fn in cfg_fns:
    if verbose:
        print "CFG: ", cfg_fn
    config, dir_name = util.load_configuration(cfg_fn)

    params = util.create_param_dict(config)

    df = pd.DataFrame(util.parseParams(params))

    """
    In [30]: for i in range(12):
    ...:     plt.figure()
    ...:     plt.plot(tr_data[:,i], 'b')
    ...:     plt.plot(test_data[:,i], 'r')
    ...:     plt.plot(pred_angles_test[:,i/4], 'g')
    ...:     plt.plot(pred_angles_tr[:,i/4], 'c')
    ...:     plt.plot(angles[:,i/4], 'm')
    ...: plt.show()
    """


    all_predY = None
    all_error = None
    mean_errors = []
    std_errors = []
    for iter_number in range(params['exp_details__num_iterations_per_setting']):
        ######## TRAINING SECTION #########


        mobiles, bases, phases = data_generation.generate_phases_from_real_data(
            np.load('../data/numpy_data/2016-9-20_day1_rawPhase_median.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-24_day3_inside_rawPhase.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-20_day1_rawPhase.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-21_day2_rawPhase.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-21_day2_rawPhase_angle.npz')['all_samples'])

        angles = data_generation.get_mobile_angles(bases, mobiles, 2)
        angles %= 360
        angles /= 360.
        #angles = np.nan_to_num(angles)

        lower_model_l = []
        lower_model_output_l = []


        #data = normalize(phases, data_min, data_max)
        #data = normalize_cols(phases)
        #data = phases
        #data = normalize_cols(phases)

        #phases_avg = np.zeros((phases.shape[0], 3))
        #for i in range(3):
        #    phases_avg[:,i] = np.mean(phases[:,i*4:(i+1)*4], axis=1)
        #data, data_min, data_max = normalize_unitAndCol(phases_avg)

        data_min = np.abs(np.min(phases, axis=0))
        data_max= np.max(phases, axis=0)
        #data, data_min, data_max = normalize_unitAndCol(phases)
        #data = normalize_unit(phases)
        #data = normalize_cols_data(phases, data_min, data_max)
        data = zero_mean(phases)
        # just a copy for plotting
        tr_data = np.copy(data)

        trainXs, trainY, testXs, testY = util.test_train_split(data, angles)

        # train the n base station phase->angle neural nets
        for i in range(params['data__num_stations']):
            # this will use all phase data from all base stations
            model_lower = models.BaseMLP(np.hstack(trainXs).shape[1], [500,50, 200,50],
                                1, epsilon=params['NN__epsilon'])

            # this model uses only one base station at a time to train - bs models
            # model_lower = models.BaseMLP(np.hstack(trainXs).shape[1]/3, [500,50],
            #                     1, epsilon=params['NN__epsilon'])


            loss = model_lower.trainModel(trainXs,
            #loss = model_lower.trainModel([trainXs[0][:,i*4:(i+1)*4]],
                                          trainY[:,i].reshape(trainY.shape[0],1),
                                          #[testXs[0][:,i*4:(i+1)*4]],
                                          testXs,
                                          testY[:,i].reshape(testY.shape[0], 1),
                                          n_epoch=params['NN__n_epochs'],
                                          batchsize=params['NN__batchsize'],
                                          max_flag=params['NN__take_max'],
                                          verbose=verbose)

            predY, error = model_lower.testModel([data.astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
            #predY, error = model_lower.testModel([data[:,i*4:(i+1)*4].astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
            print "Phase->Angle NN Error: ", np.mean(error)

            # save angle predictions
            lower_model_output_l.append(predY)

            # save each lower level model
            lower_model_l.append(model_lower)

        # combine all lower level angle predictions for all the base stations
        #for use as input to upper layer nn
        pred_angles = np.hstack(lower_model_output_l)
        pred_angles_tr = np.copy(pred_angles)

        # replicate data for input to SMLP
        #trainXs, trainY, testXs, testY = util.test_train_split(pred_angles, mobiles)
        trainXs, trainY, testXs, testY = util.test_train_split(angles, mobiles)

        # upper layer NN model
        # model = models.NBPStructuredMLP(trainXs[0].shape[1], params['NN__network_size'][0],
        #                                     params['NN__network_size'][1], params['data__ndims'],
        #                                     len(rep_idxs), epsilon=params['NN__epsilon'])

        model= models.BaseMLP(trainXs[0].shape[1], [500,50, 200,50],
                            2, epsilon=params['NN__epsilon'])

        loss = model.trainModel(trainXs, trainY, testXs, testY,
                                    n_epoch=params['NN__n_epochs'],
                                    batchsize=params['NN__batchsize'],
                                    max_flag=params['NN__take_max'],
                                    verbose=verbose)

        _ , error = model.testModel(testXs, testY)
        print "Training error upper: ", np.mean(error)

        f = open(dir_name + 'loss_iteration%d.txt' % (iter_number), 'w')
        f.write("%f" % (loss))
        f.close()

        print "loss: ", loss


        ######## TEST SECTION #########

        # generate mobile points, base stations, and angles
        mobiles, bases, phases = data_generation.generate_phases_from_real_data(
            np.load('../data/numpy_data/2016-9-21_day2_rawPhase_median.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-25_day4_inside_rawPhase.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-21_day2_rawPhase.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-20_day1_rawPhase_median.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-20_day1_rawPhase.npz')['all_samples'])
            #np.load('../data/numpy_data/2016-9-20_day1_rawPhase_angle.npz')['all_samples'])

        #data = normalize(phases, data_min, data_max)
        #data = phases
        #data = normalize_unitAndCol(phases, data_min, data_max)
        # phases_avg = np.zeros((phases.shape[0], 3))
        # for i in range(3):
        #     phases_avg[:,i] = np.mean(phases[:,i*4:(i+1)*4], axis=1)
        # data, _, _ = normalize_unitAndCol(phases_avg, data_min, data_max)

        #data, _, _ = normalize_unitAndCol(phases, data_min, data_max)
        #data = normalize_unit(phases)
        #data = normalize_cols_data(phases, data_min, data_max)
        data = zero_mean(phases)
        test_data = np.copy(data)


        # get angle by running phases through all of the lower level NNs
        # trainXs, trainY, testXs, testY = util.test_train_split(data, angles, 0.)
        test_lower_model_output_l = []


        for i,m in enumerate(lower_model_l):
            predY, error = m.testModel([data.astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
            # predY, error = model_lower.testModel([data[:,i*4:(i+1)*4].astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))

            test_lower_model_output_l.append(predY)
            print "Test Phase2Ang error: ", np.mean(error)

        # combine all lower level angle predictions for all the base stations
        #for use as input to upper layer nn
        pred_angles= np.hstack(test_lower_model_output_l)
        pred_angles_test = np.copy(pred_angles)

        trainXs, trainY, testXs, testY = util.test_train_split(pred_angles, mobiles, 0.)

        # test model
        predY, error = model.testModel(testXs, testY)

        f = open(dir_name + 'error_iteration%d.txt' % iter_number, 'w')
        f.write("Mean Error: %f\n" % (np.mean(error)))
        f.write("Error Standard Deviation: %f\n" % (np.std(error)))
        f.close()

        mean_errors.append(np.mean(error))
        std_errors.append(np.std(error))

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

    f = open(dir_name + 'resultsdata.npz', 'w')
    np.savez(f, all_predY=all_predY, all_error=all_error)
    f.close()

    df['mean_err'] = np.mean(mean_errors)
    df['std_err'] = np.mean(std_errors)
    df_all = df_all.append(df, ignore_index=True)

    print np.mean(mean_errors)

    # print out warning if figures not saved
    if params['exp_details__save']:
        print "****** Figures saved to directory %s ********" % (dir_name)
    else:
        print "****** Not saving!!!! ****** "
        print "If you would like to save, change the config file"

res_dir_folder = 'aggregated_results'
if not os.path.exists(res_dir_folder):
    os.makedirs(res_dir_folder)

res_dir_name = "%s/%s__%s.csv" % (res_dir_folder, config.get("exp_details", "setname"), datetime.datetime.now().strftime("%m_%d_%Y_%I:%M:%S%p"))
df_all.to_csv(res_dir_name)
