import ConfigParser
import json
import datetime
import os
import shutil
import numpy as np
import ast

# TODO: convert output to dictionary
def load_configuration(cfg_fn):
    config = ConfigParser.ConfigParser()

    config.read(cfg_fn)

    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")

    dir_name = "experiment_results/%s__%s/" % (config.get("exp_details", "name"), datetime.datetime.now().strftime("%m_%d_%Y_%I:%M:%S%p"))


    if ast.literal_eval(config.get("exp_details", "save")):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # copy cfg file to experiment directory for record keeping purposes
        shutil.copy(cfg_fn, dir_name)

    return config, dir_name

    # print config.sections()
    # print config.get("NN", "type")
    # print json.loads(config.get("NN", "network_size"))
    # print config.get("exp_details", "name")


def test_train_split(X, Y, training_size=0.8):
    # X = X/360.
    trainXs, testXs  = [], []

    trainXs.append( X[:X.shape[0]*training_size,:].astype(np.float32))
    testXs.append( X[X.shape[0]*training_size:,:].astype(np.float32))

    # Could be a list??? Comiter thinks so....
    trainY = Y[:Y.shape[0]*training_size,:].astype(np.float32)
    testY = Y[Y.shape[0]*training_size:,:].astype(np.float32)

    return trainXs, trainY, testXs, testY


def parseParams(params):
    new_params = {}
    for param,_ in params.iteritems():
        if type(params[param]) == list:
            new_params[param] = [params[param].__str__()]
        elif type(params[param]) == dict:
            for k,v in params[param].iteritems():
                if type(v) == list:
                    new_v = v.__str__()
                else:
                    new_v = v
                new_params[k] = [new_v]
        else:
            new_params[param] = [params[param]]

    return new_params


def create_param_dict(config):
    params = {}
    params['NN__type'] = config.get("NN", "type")
    params['NN__network_size'] = json.loads(config.get("NN", "network_size"))
    params['NN__n_epochs'] = int(config.get("NN", "n_epochs"))
    params['NN__batchsize'] = int(config.get("NN", "batchsize"))
    params['NN__take_max'] = ast.literal_eval(config.get("NN", "take_max"))
    params['data__num_pts'] = int(config.get("data", "num_pts"))
    params['data__ndims'] = int(config.get("data", "ndims"))
    params['data__num_stations'] = int(config.get("data", "num_stations"))
    params['data__bs_type'] = config.get("data", "bs_type")
    params['exp_details__name'] = config.get("exp_details", "name")
    params['exp_details__description'] = config.get("exp_details", "description")
    params['exp_details__save'] = ast.literal_eval(config.get("exp_details", "save"))
    params['exp_details__interactive'] = ast.literal_eval(config.get("exp_details", "interactive"))
    params['exp_details__num_iterations_per_setting'] = ast.literal_eval(config.get("exp_details", "num_iterations_per_setting"))
    params['noise__addnoise_train'] = ast.literal_eval(config.get("noise", "addnoise_train"))
    params['noise__addnoise_test'] = ast.literal_eval(config.get("noise", "addnoise_test"))
    if params['noise__addnoise_train'] or params['noise__addnoise_test']:
        params['noise__noise_model'] = config.get("noise", "noise_model")
        params['noise__noise_params'] = ast.literal_eval(config.get("noise", "noise_params"))
        params['noise__bases_to_noise'] = json.loads(config.get("noise", "bases_to_noise"))

    return params



