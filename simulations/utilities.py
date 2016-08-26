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
	
	dir_name = "experiment_results/%s__%s/" % (config.get("exp_details", "name"), datetime.datetime.now().strftime("%m_%d_%Y_%I:%M%p"))
	

	if ast.literal_eval(config.get("exp_details", "save")):
		if not os.path.exists(dir_name):
		    os.makedirs(dir_name)
		
		# copy cfg file to experiment directory for record keeping purposes
		shutil.copy(cfg_fn, dir_name)
	
	return config

	# print config.sections()
	# print config.get("NN", "type")
	# print json.loads(config.get("NN", "network_size"))
	# print config.get("exp_details", "name")


def test_train_split(X, Y, training_size=0.8):
	X = X/360.
	trainXs, testXs  = [], []

	trainXs.append( X[:X.shape[0]*training_size,:].astype(np.float32))
	testXs.append( X[X.shape[0]*training_size:,:].astype(np.float32))

	# Could be a list??? Comiter thinks so....
	trainY = Y[:Y.shape[0]*training_size,:].astype(np.float32)
	testY = Y[Y.shape[0]*training_size:,:].astype(np.float32)

	return trainXs, trainY, testXs, testY


def create_param_dict(config):
	params = {}
	params['NN__type'] = config.get("NN", "type")
	params['NN__network_size'] = json.loads(config.get("NN", "network_size"))
	params['data__num_pts'] = int(config.get("data", "num_pts"))
	params['data__ndims'] = int(config.get("data", "ndims"))
	params['data__num_stations'] = int(config.get("data", "num_stations"))
	params['data__bs_type'] = config.get("data", "bs_type")
	params['exp_details__save'] = ast.literal_eval(config.get("exp_details", "save"))


