import numpy as np


def add_gaussian_noise(data, ndim,  base_idxs=[-1], noise_type="gaussian", noise_params={'mean': 0, 'std':1}):

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

    return data


def add_angle_dependent_noise():
    print "not implemented"
    return


def add_spurious_noise():
    print "not implemented"
    return


def add_wrong_noise():
    print "not implemented"
    return


def add_no_output_noise():
    print "not implemented"
    return

    