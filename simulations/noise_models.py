import numpy as np

# adds noise to the data from the given distribution (default is gaussian N(0,1))
def add_distribution_noise(data, ndim,  base_idxs=[-1], noise_type="gaussian", noise_params={'mean': 0, 'std':1}):

    if noise_type == "gaussian":
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


def add_angle_dependent_noise():
    print "not implemented"
    return


def add_spurious_noise():
    print "not implemented"
    return

# replaces the data with nonsense random noise from the given distribution (default is uniform [-1,1])
def transform_to_nonsense_noise(data, ndim,  base_idxs=[-1], noise_type="uniform", noise_params={'lower_bound': -1, 'upper_bound':1}):
    if noise_type == "uniform"
      if ndim == 2:
          # using 1 angle for each base station (alpha)
          col_idxs = base_idxs
          gauss_noise = np.random.uniform(low=noise_params['lower_bound'],\
                                     high=noise_params['upper_bound'],\
                                     size=(data.shape[0],len(col_idxs)))

          data[:,col_idxs] += gauss_noise
      
      elif ndim == 3:
          # using 3 angles for each base station (alpha beta gamma)
          for idx in base_idxs:
              gauss_noise = np.random.uniform(lower=noise_params['lower_bound'],\
                                     high=noise_params['upper_bound'],\
                                     size=(data.shape[0],ndim))
              data[:, (idx*3):(idx*3)+3] += gauss_noise
    
    else:
        assert False, "This type of transform to nonsense noise has not been implemented in noise_models.py"

    return data


def add_no_output_noise(data, ndim,  base_idxs=[-1], constant_val=0):
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

