import numpy as np

def add_noise_dispatcher(angles, noise_model, ndim, base_idxs=[-1], noise_params=None):
    if noise_model == 'add_distribution_noise':
        noisy_angles = add_distribution_noise(angles, ndim, base_idxs=base_idxs,
                                                    noise_params=noise_params )

    elif noise_model == 'add_angle_dependent_noise':
        noisy_angles = add_angle_dependent_noise(angles, ndim, base_idxs=base_idxs,
                                                    noise_params=noise_params )

    elif noise_model == 'add_spurious_noise':
        noisy_angles = add_spurious_noise(angles, ndim, base_idxs=base_idxs,
                                                    noise_params=noise_params )

    elif noise_model == 'transform_to_nonsense_noise':
        noisy_angles = transform_to_nonsense_noise(angles, ndim, base_idxs=base_idxs,
                                                    noise_params=noise_params )

    elif noise_model == 'add_no_output_noise':
        noisy_angles = add_no_output_noise(angles, ndim, base_idxs=base_idxs,
                                                    noise_params=noise_params )

    else:
        assert False, "There is no noise model matching the one you have selected in the config file"

    return noisy_angles


# adds noise to the data from the given distribution (default is gaussian N(0,1))
def add_distribution_noise(data, ndim,  base_idxs=[-1], noise_params={'noise_type': 'gaussian', 'mean': 0, 'std':1}):

    if noise_params['noise_type'] == "gaussian":
        if ndim == 2:
            # using 1 angle for each base station (alpha)
            col_idxs = base_idxs
            gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                       scale=noise_params['std'],\
                                       size=(data.shape[0],len(col_idxs)))

            data[:,col_idxs] += gauss_noise
        
        elif ndim == 3:
            # using 3 angles for each base station (alpha beta gamma)
            for idx in base_idxs:
                gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                       scale=noise_params['std'],\
                                       size=(data.shape[0],ndim))
                data[:, (idx*3):(idx*3)+3] += gauss_noise

    else:
        assert False, "This type of addative noise has not been implemented in noise_models.py"


    return data


# nonlinear function for add_angle_dependent_noise
def nonlinear_effect_function(data, k, j):
    multiplier_data = np.exp(np.abs(k * (data - (90/360.)))) - j
    return multiplier_data


# adds noise based on the angle of the point to the given base station
def add_angle_dependent_noise(data, ndim,  base_idxs=[-1],  noise_params={'noise_type': 'gaussian', 'k': 2, 'j': 1, 'mean': 0, 'std':1}):
    if noise_params['noise_type'] == "gaussian":
        if ndim == 2:
            # using 1 angle for each base station (alpha)
            col_idxs = base_idxs
            gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                       scale=noise_params['std'],\
                                       size=(data.shape[0],len(col_idxs)))

            data_multiplier = nonlinear_effect_function(data, noise_params['k'], noise_params['j'])

            gauss_noise = np.multiply(gauss_noise, data_multiplier)

            data[:,col_idxs] += gauss_noise
        
        elif ndim == 3:
            # using 3 angles for each base station (alpha beta gamma)
            for idx in base_idxs:
                gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                       scale=noise_params['std'],\
                                       size=(data.shape[0],ndim))

                data_multiplier = nonlinear_effect_function(data, noise_params['k'], noise_params['j'])

                gauss_noise = np.multiply(gauss_noise, data_multiplier)

                data[:, (idx*3):(idx*3)+3] += gauss_noise

    else:
        assert False, "This type of addative noise has not been implemented in noise_models.py"


    return data


def add_spurious_noise(data, ndim,  base_idxs=[-1], noise_params={'noise_type': "uniform", 'corruption_rate': 0.1, 'lower_bound': -1, 'upper_bound':1}):
    if noise_params['noise_type'] == 'uniform':
        if ndim == 2:
            # using 1 angle for each base station (alpha)
            col_idxs = base_idxs
            row_idxs = np.random.permutation(data.shape[0])[:data.shape[0]*noise_params['corruption_rate']]
            row_idxs = np.sort(row_idxs)

            noise = np.random.uniform(low=noise_params['lower_bound'],\
                                           high=noise_params['upper_bound'],\
                                           size=(len(row_idxs),len(col_idxs)))

            # row_idxs = list(row_idxs)
            col_idxs = np.array(col_idxs)
            row_idxs = np.tile(row_idxs, col_idxs.shape[0])
            col_idxs = np.tile(col_idxs, row_idxs.shape[0]/col_idxs.shape[0])
            data[row_idxs,col_idxs] = noise.ravel()
        
        elif ndim == 3:
            # using 3 angles for each base station (alpha beta gamma)
            for idx in base_idxs:
                row_idxs = np.random.permutation(data.shape[0])[:data.shape[0]*noise_params['corruption_rate']]
                noise = np.random.uniform(low=noise_params['lower_bound'],\
                                           high=noise_params['upper_bound'],\
                                           size=(len(row_idxs),len(col_idxs)))
                data[:, (idx*3):(idx*3)+3] = noise
    else:
        assert False, "This type of transform to nonsense noise has not been implemented in noise_models.py"



    return data


# replaces the data with nonsense random noise from the given distribution (default is uniform [-1,1])
def transform_to_nonsense_noise(data, ndim,  base_idxs=[-1], noise_params={'noise_type': "uniform", 'lower_bound': -1, 'upper_bound':1}):
    if noise_params['noise_type'] == "uniform":
        if ndim == 2:
            # using 1 angle for each base station (alpha)
            col_idxs = base_idxs
            noise = np.random.uniform(low=noise_params['lower_bound'],\
                                       high=noise_params['upper_bound'],\
                                       size=(data.shape[0],len(col_idxs)))

            data[:,col_idxs] = noise
        
        elif ndim == 3:
            # using 3 angles for each base station (alpha beta gamma)
            for idx in base_idxs:
                noise = np.random.uniform(lower=noise_params['lower_bound'],\
                                       high=noise_params['upper_bound'],\
                                       size=(data.shape[0],ndim))
                data[:, (idx*3):(idx*3)+3] = noise
    
    else:
        assert False, "This type of transform to nonsense noise has not been implemented in noise_models.py"


    return data


def add_no_output_noise(data, ndim,  base_idxs=[-1], noise_params={'constant_val':0}):
    if ndim == 2:
        # using 1 angle for each base station (alpha)
        col_idxs = base_idxs
        zeros = np.zeros((data.shape[0],len(col_idxs))) + constant_val
        data[:,col_idxs] = zeros

    elif ndim == 3:
        # using 3 angles for each base station (alpha beta gamma)
        for idx in base_idxs:
            zeros = np.zeros((data.shape[0],ndim)) + constant_val
            data[:, (idx*3):(idx*3)+3] = zeros


    return data






