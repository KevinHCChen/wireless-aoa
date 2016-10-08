import numpy as np
import copy

def add_noise_dispatcher(angles, mobiles, noise_model, ndim, base_idxs=[-1], noise_params=None, ):
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

    elif noise_model == 'add_multipath_noise':
        noisy_angles, mobiles = add_multipath_noise(angles, ndim, mobiles, base_idxs=base_idxs)
                                                    #noise_params=noise_params )


    else:
        assert False, "There is no noise model matching the one you have selected in the config file"

    return noisy_angles, mobiles


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
    mult_data_copy = copy.deepcopy(data)

    mult_data_copy[np.where(mult_data_copy>180/360.)] = mult_data_copy[np.where(mult_data_copy>180/360.)] - 0.5
    multiplier_data = np.exp(np.abs(k * (mult_data_copy - (90/360.)))) - j

    return multiplier_data


# adds noise based on the angle of the point to the given base station
def add_angle_dependent_noise(data, ndim,  base_idxs=[-1],  noise_params={'noise_type': 'gaussian', 'k': 4, 'j': 1, 'mean': 0, 'std':0.01}):
    if noise_params['noise_type'] == "gaussian":
        if ndim == 2:
            # using 1 angle for each base station (alpha)
            col_idxs = base_idxs
            gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                       scale=noise_params['std'],\
                                       size=(data.shape[0],len(col_idxs)))

            data_multiplier = nonlinear_effect_function(data[:,col_idxs], noise_params['k'], noise_params['j'])

            gauss_noise = np.multiply(gauss_noise, data_multiplier)
            assert  gauss_noise.shape == data[:,col_idxs].shape, "shapes don't match - whoops!"
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
        zeros = np.zeros((data.shape[0],len(col_idxs))) + noise_params['constant_val']
        data[:,col_idxs] = zeros

    elif ndim == 3:
        # using 3 angles for each base station (alpha beta gamma)
        for idx in base_idxs:
            zeros = np.zeros((data.shape[0],ndim)) + noise_params['constant_val']
            data[:, (idx*3):(idx*3)+3] = zeros


    return data


def add_multipath_noise(data, ndim, mobiles,  base_idxs=[-1], noise_params={'mp_regions': [[(-2,-2),(-1,-1)],[(1,-2),(2,-1)], [(1,1),(2,2)]] }):

    noise_vals = [10., 20.]#, 30., 40., 5.]
    base_stat_idxs = [0,2]
    if ndim == 2:
        # using 1 angle for each base station (alpha)

        # print mobiles.shape
        # print angles.shape
        # print mobiles
        # print angles
        # assert False
        original_size = data.shape[0]

        mobiles = np.tile(mobiles,(len(noise_params['mp_regions'])*len(noise_vals)+1,1))
        data = np.tile(data,(len(noise_params['mp_regions'])*len(noise_vals)+1,1))

        print mobiles.shape, data.shape


        for i, mp_region in enumerate(noise_params['mp_regions']):
            x0 = mp_region[0][0]
            y0 = mp_region[0][1]
            xn = mp_region[1][0]
            yn = mp_region[1][1]

            toaddidxs = np.where(((mobiles[:,0] < xn) & (mobiles[:,0] > x0)) & ((mobiles[:,1] < yn) & (mobiles[:,1] > y0)))[0]


            for j, nv in enumerate(noise_vals):
                toaddidxs_t = toaddidxs[np.where(toaddidxs <= original_size)]
                print "%d Points Affected by Multipath" % (len(toaddidxs_t))
                toaddidxs_t += ((i+1)+(j*len(noise_params['mp_regions'])))*original_size
                #print toaddidxs_t
                # data[toaddidxs, np.random.random_integers(0, high=mobiles.shape[1]-1)] += 70/360.#np.random.random_integers(0, high=360, size=1)/360.
                #data[toaddidxs_t, 0] += nv/360.#np.random.random_integers(0, high=360, size=1)/360.
                data[toaddidxs_t, 0] += nv/360.#np.random.random_integers(0, high=360, size=1)/360.
                #print data[toaddidxs_t, 0]


    elif ndim == 3:
        assert False, "Not implemented yet"
        # using 3 angles for each base station (alpha beta gamma)
        for idx in base_idxs:
            gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                   scale=noise_params['std'],\
                                   size=(data.shape[0],ndim))
            data[:, (idx*3):(idx*3)+3] += gauss_noise


    data %= 1
    return data, mobiles



