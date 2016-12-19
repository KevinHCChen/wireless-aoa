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
parser.add_argument('--datasetfn1', '-a', dest='datasetfn1', type=str,
                    help='Which dataset 1 file name to use')
parser.add_argument('--datasetfn2', '-b', dest='datasetfn2', type=str,
                    help='Which dataset 2 file name to use')
args = parser.parse_args()

showfig = args.showfig
configfile = args.configfile
startidx = args.startidx
endidx = args.endidx
configfile_dir = args.configfile_dir
startidx = args.startidx
endidx = args.endidx
verbose = args.verbose
datasetfn1 = args.datasetfn1
datasetfn2 = args.datasetfn2

import utilities as util
import models as models
import data_generation as data_generation
import plotting as plotting

num_antennas_per_bs = 4
used_antennas_per_bs = 4
use_dir = False 


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


    all_predY = None
    all_error = None
    mean_errors = []
    std_errors = []
    for iter_number in range(params['exp_details__num_iterations_per_setting']):
        ######## TRAINING SECTION #########
        # parsed_data1 = np.load(datasetfn1)

        

        mobiles, bases, phases = data_generation.generate_phases_from_real_data(np.load('../data/numpy_data/2016-9-24_day3_inside_rawPhase.npz')['all_samples'])
        # mobiles, bases, angles = data_generation.generate_from_real_data(np.load('../data/numpy_data/2016-9-20_day1.npz')['all_samples'])


        num_repeat_for_more_data_please_work = 20

        additional_copies_phases = np.repeat(phases, num_repeat_for_more_data_please_work, axis=0)
        additional_copies_phases += np.random.normal(loc=0, scale=0.05, size=additional_copies_phases.shape)

        mobiles = np.repeat(mobiles, num_repeat_for_more_data_please_work, axis=0)

        phases = additional_copies_phases


        phases = zero_mean(phases)

        print "Angles Real 1##########"
        # print angles[:10]
        print phases[:10]
        angles = data_generation.get_mobile_angles(bases, mobiles, 2)
        angles %= 360
        angles /= 360.
        angles = np.nan_to_num(angles)
        print "Angles Computer 1##########"
        print angles[:10]

        # angles = np.repeat(angles, num_repeat_for_more_data_please_work, axis=0)

        # generate mobile points, base stations, and angles
        # mobiles, bases, angles, phases = data_generation.generate_data(params['data__num_pts'],
        #                                                        params['data__num_stations'],
        #                                                        params ['data__ndims'],
        #                                                        pts_r=3, bs_r=4,
        #                                                        bs_type=params['data__bs_type'],
        #                                                        points_type=params['data__data_dist'])
        # print angles.shape
        # print phases.shape
        # assert False

        # IMPORTANT: remember to add noise before replicating data (e.g., for snbp-mlp)
        if params['noise__addnoise_train']:
            angles, phases, mobiles = noise_models.add_noise_dispatcher(angles, phases, mobiles, params['data__data_type'],
                                                       params['noise__noise_model'],
                                                       params['data__ndims'],
                                                       base_idxs=params['noise__bases_to_noise'],
                                                       noise_params=params['noise__noise_params'])



        if params['NN__type'] == 'snbp-mlp' or params['NN__type'] == 'smlp':
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            print rep_idxs
            if params['data__data_type'] == 'phases':
                print "************PHASES************"
                tmp = []
                for x1,x2 in rep_idxs:
                    tmp.append(tuple(range(x1*3,(x1*3)+3)) + tuple(range(x2*3,(x2*3)+3)))



                rep_idxs = tmp
                #[(rep_idx1*(num_antennas_per_bs-1) + i, rep_idx2*(num_antennas_per_bs-1) + i) for rep_idx1, rep_idx2 in rep_idxs for i in range(num_antennas_per_bs-1)]
                data = data_generation.replicate_data(phases, params['data__ndims'],  rep_idxs)
            elif params['data__data_type'] == 'angles':
                data = data_generation.replicate_data(angles, params['data__ndims'],  rep_idxs)
        elif params['NN__type'] == 'bmlp':
            if params['data__data_type'] == 'phases':
                data = phases  
            elif params['data__data_type'] == 'angles':
                data = angles

        data = phases

        # for phase offset SMLP
        '''
        if params['NN__type'] == 'phased-snbp-mlp' or params['NN__type'] == 'smlp':
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            rep_idxs = [base_idx*(num_antennas_per_bs-1) + i for base_idx in rep_idxs for i in range(num_antennas_per_bs-1)]
            angles = data_generation.replicate_data(phases, params['data__ndims'],  rep_idxs)
        '''


        # split data
        #trainXs, trainY, testXs, testY = util.test_train_split(data, mobiles)
        #trainXs, trainY, testXs, testY = util.test_train_split(data, np.matrix(angles[:,0]).T)
        #angles = angles[:,0].reshape(angles.shape[0],1)
        #trainXs, trainY, testXs, testY = util.test_train_split(data, angles)


        lower_model_l = []
        lower_model_output_l = []

        if params['NN__type'] == "bmlp":
            model = models.BaseMLP(np.hstack(trainXs).shape[1], params['NN__network_size'],
                                   1, epsilon=params['NN__epsilon'])
                                   #params['data__ndims'], epsilon=params['NN__epsilon'])
        elif params['NN__type'] == 'smlp':
            model = models.StructuredMLP(None, params['NN__network_size'][0],
                                         params['NN__network_size'][1], params['data__ndims'],
                                         rep_idxs)
        elif params['NN__type'] == 'snbp-mlp':
            model = models.NBPStructuredMLP(trainXs[0].shape[1], params['NN__network_size'][0],
                                            params['NN__network_size'][1], params['data__ndims'],
                                            len(rep_idxs), epsilon=params['NN__epsilon'])
        
        elif params['NN__type'] == 'e2e-nostructure':
            trainXs, trainY, testXs, testY = util.test_train_split(data, mobiles)

            # upper layer NN model
            model = models.BaseMLP(np.hstack(trainXs).shape[1], params['NN__network_size'],
                                   params['data__ndims'], epsilon=params['NN__epsilon'])

            loss = model.trainModel(trainXs, trainY, testXs, testY,
                                       n_epoch=params['NN__n_epochs'],
                                       batchsize=params['NN__batchsize'],
                                       max_flag=params['NN__take_max'],
                                       verbose=verbose)

            f = open(dir_name + 'loss_iteration%d.txt' % (iter_number), 'w')
            f.write("%f" % (loss))
            f.close()
        elif params['NN__type'] == 'e2e':
            trainXs, trainY, testXs, testY = util.test_train_split(data, angles, 0.99)

            # train the n base station phase->angle neural nets
            for i in range(params['data__num_stations']):
                #model_lower = models.BaseMLP(np.hstack(trainXs).shape[1]/3, [200,50],
                model_lower = models.BaseMLP(np.hstack(trainXs).shape[1], [500,50, 200,50],
                                    1, epsilon=params['NN__epsilon'])

                #loss = model_lower.trainModel([trainXs[0][:,i*3:(i+1)*3]], trainY[:,i].reshape(trainY.shape[0],1), [testXs[0][:,i*3:(i+1)*3]], testY[:,i].reshape(testY.shape[0], 1),
                loss = model_lower.trainModel(trainXs, trainY[:,i].reshape(trainY.shape[0],1), testXs, testY[:,i].reshape(testY.shape[0], 1),
                                    n_epoch=params['NN__n_epochs'],
                                    batchsize=params['NN__batchsize'],
                                    max_flag=params['NN__take_max'],
                                    verbose=verbose)

                # MC says note cheating :) ....
                #predY, error = model_lower.testModel([data[:,(i*3):(i+1)*3].astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
                predY, error = model_lower.testModel([data.astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
                print "ERRORS [Pred, Real]: \n ", np.hstack((predY[:20],angles[:,i].reshape(angles.shape[0], 1).astype(np.float32)[:20]))
                print "Phase->Angle NN Error: ", np.mean(error)

                # save angle predictions
                lower_model_output_l.append(predY)

                # save each lower level model
                lower_model_l.append(model_lower)

            # combine all lower level angle predictions for all the base stations for use as input to upper layer nn
            pred_angles = np.hstack(lower_model_output_l)

            # replicate data for input to SMLP
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            pred_angles = data_generation.replicate_data(pred_angles, params['data__ndims'],  rep_idxs)

            trainXs, trainY, testXs, testY = util.test_train_split(pred_angles, mobiles, 0.99)

            # upper layer NN model
            model = models.NBPStructuredMLP(trainXs[0].shape[1], params['NN__network_size'][0],
                                                 params['NN__network_size'][1], params['data__ndims'],
                                                len(rep_idxs), epsilon=params['NN__epsilon'])

            loss = model.trainModel(trainXs, trainY, testXs, testY,
                                       n_epoch=params['NN__n_epochs'],
                                       batchsize=params['NN__batchsize'],
                                       max_flag=params['NN__take_max'],
                                       verbose=verbose)

            f = open(dir_name + 'loss_iteration%d.txt' % (iter_number), 'w')
            f.write("%f" % (loss))
            f.close()

            predY, error = model.testModel(testXs, testY)
            print "Training ERROR: ", np.mean(error)
            print "Training [Pred, Real]\n", np.hstack((predY, testY)) 
            

        elif params['NN__type'] == 'e2e-allstructured':
            # create rep_idxs on a per-base station unit manner
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            # expand rep_idxs to be on an antenna unit manner
            tmp = []
            for x1,x2 in rep_idxs:
                tmp.append(tuple(range(x1*used_antennas_per_bs,(x1*used_antennas_per_bs)+used_antennas_per_bs)) + tuple(range(x2*used_antennas_per_bs,(x2*used_antennas_per_bs)+used_antennas_per_bs)))
            rep_idxs = tmp

            # replicate data
            data = data_generation.replicate_data(data, params['data__ndims'],  rep_idxs)

            print "Data shape: ", data.shape
            trainXs, trainY, testXs, testY = util.test_train_split(data, angles)

            print "trainXs shape: ", trainXs[0].shape

            # train the n base station phase->angle SMLPs
            for i in range(params['data__num_stations']):
                #model_lower = models.BaseMLP(np.hstack(trainXs).shape[1]/3, [200,50],


                model_lower = models.NBPStructuredMLP(trainXs[0].shape[1], params['NN__network_size'][0],
                                                params['NN__network_size'][1], 1,
                                                len(rep_idxs), unit_per_bs=4, epsilon=params['NN__epsilon'])
                            

                #loss = model_lower.trainModel([trainXs[0][:,i*3:(i+1)*3]], trainY[:,i].reshape(trainY.shape[0],1), [testXs[0][:,i*3:(i+1)*3]], testY[:,i].reshape(testY.shape[0], 1),
                loss = model_lower.trainModel(trainXs, trainY[:,i].reshape(trainY.shape[0],1), testXs, testY[:,i].reshape(testY.shape[0], 1),
                                    n_epoch=params['NN__n_epochs'],
                                    batchsize=params['NN__batchsize'],
                                    max_flag=params['NN__take_max'],
                                    verbose=verbose)

                # MC says note cheating :) ....
                #predY, error = model_lower.testModel([data[:,(i*3):(i+1)*3].astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
                predY, error = model_lower.testModel([data.astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
                print "Phase->Angle NN Error: ", np.mean(error)

                # save angle predictions
                lower_model_output_l.append(predY)

                # save each lower level model
                lower_model_l.append(model_lower)

            # combine all lower level angle predictions for all the base stations for use as input to upper layer nn
            pred_angles = np.hstack(lower_model_output_l)

            # replicate data for input to SMLP
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            pred_angles = data_generation.replicate_data(pred_angles, params['data__ndims'],  rep_idxs)

            trainXs, trainY, testXs, testY = util.test_train_split(pred_angles, mobiles)

            # upper layer NN model
            model = models.NBPStructuredMLP(trainXs[0].shape[1], params['NN__network_size'][0],
                                                params['NN__network_size'][1], params['data__ndims'],
                                                len(rep_idxs), epsilon=params['NN__epsilon'])

            loss = model.trainModel(trainXs, trainY, testXs, testY,
                                       n_epoch=params['NN__n_epochs'],
                                       batchsize=params['NN__batchsize'],
                                       max_flag=params['NN__take_max'],
                                       verbose=verbose)

            f = open(dir_name + 'loss_iteration%d.txt' % (iter_number), 'w')
            f.write("%f" % (loss))
            f.close()





        ######## TEST SECTION #########

        # generate mobile points, base stations, and angles
        # mobiles, bases, angles, phases = data_generation.generate_data(50*50,
        #                                                        params['data__num_stations'],
        #                                                        params ['data__ndims'],
        #                                                        pts_r=3, bs_r=4,
        #                                                        bs_type=params['data__bs_type'], points_type="grid")
        mobiles, bases, phases = data_generation.generate_phases_from_real_data(np.load('../data/numpy_data/2016-9-25_day4_inside_rawPhase.npz')['all_samples'])
        # mobiles, bases, angles = data_generation.generate_from_real_data(np.load('../data/numpy_data/2016-9-21_day2.npz')['all_samples'])

        phases = zero_mean(phases)


        print "Angles Real 2##########"
        print angles[:10]
        print phases[:10]
        angles = data_generation.get_mobile_angles(bases, mobiles, 2)
        angles %= 360
        angles /= 360.
        angles = np.nan_to_num(angles)
        print "Angles Computer 2##########"
        print angles[:10]

        if params['noise__addnoise_test']:
            angles, phases, mobiles = noise_models.add_noise_dispatcher(angles, phases, mobiles, params['data__data_type'],
                                                                params['noise__noise_model'],
                                                                params['data__ndims'],
                                                                base_idxs=params['noise__bases_to_noise'],
                                                                noise_params=params['noise__noise_params'])


        if params['NN__type'] == 'snbp-mlp' or params['NN__type'] == 'smlp':
            if params['data__data_type'] == 'phases':
                data = data_generation.replicate_data(phases, params['data__ndims'],  rep_idxs)
            elif params['data__data_type'] == 'angles':
                data = data_generation.replicate_data(angles, params['data__ndims'],  rep_idxs)
        elif params['NN__type'] == 'bmlp':
            if params['data__data_type'] == 'phases':
                data = phases
            elif params['data__data_type'] == 'angles':
                data = angles

        data = phases


        # get angle by running phases through all of the lower level NNs
        # trainXs, trainY, testXs, testY = util.test_train_split(data, angles, 0.)
        test_lower_model_output_l = []

        
        if params['NN__type'] == 'e2e-allstructured':
            # create rep_idxs on a per-base station unit manner
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            # expand rep_idxs to be on an antenna unit manner
            tmp = []
            for x1,x2 in rep_idxs:
                tmp.append(tuple(range(x1*used_antennas_per_bs,(x1*used_antennas_per_bs)+used_antennas_per_bs)) + tuple(range(x2*used_antennas_per_bs,(x2*used_antennas_per_bs)+used_antennas_per_bs)))
            rep_idxs = tmp

            # replicate data
            data = data_generation.replicate_data(data, params['data__ndims'],  rep_idxs)

        

        if params['NN__type'] == 'e2e' or params['NN__type'] == 'e2e-allstructured':
            for i,m in enumerate(lower_model_l):
                # MC says not cheating :) ....
                #predY, error = m.testModel(testXs, testY[:,i].reshape(testY.shape[0],1))
                predY, error = m.testModel([data.astype(np.float32)], angles[:,i].reshape(angles.shape[0], 1).astype(np.float32))
                # print "PRED Y 2:", predY[:20]
                # print "ACTUAL 2:", angles[:,i].reshape(angles.shape[0], 1).astype(np.float32)[:20]
                print "ERRORS [Pred, Real]: \n ", np.hstack((predY[:20],angles[:,i].reshape(angles.shape[0], 1).astype(np.float32)[:20]))
                print "Testing Phase->Angle NN Error: ", np.mean(error)
                test_lower_model_output_l.append(predY)

            # combine all lower level angle predictions for all the base stations for use as input to upper layer nn
            pred_angles = np.hstack(test_lower_model_output_l)

            # replicate data for input to SMLP
            rep_idxs = [comb for comb in itertools.combinations(range(params['data__num_stations']),2)]
            pred_angles = data_generation.replicate_data(pred_angles, params['data__ndims'],  rep_idxs)

            trainXs, trainY, testXs, testY = util.test_train_split(pred_angles, mobiles, 0.)

            # test model
            predY, error = model.testModel(testXs, testY)

            print "Test ERROR: ", np.mean(error)
            print "Test [Pred, Real]\n", np.hstack((predY, testY)) 
            # assert False

            f = open(dir_name + 'error_iteration%d.txt' % iter_number, 'w')
            f.write("Mean Error: %f\n" % (np.mean(error)))
            f.write("Error Standard Deviation: %f\n" % (np.std(error)))
            f.close()

        if params['NN__type'] == 'e2e-nostructure':
            trainXs, trainY, testXs, testY = util.test_train_split(data, mobiles, 0.)
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

    assert False

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
