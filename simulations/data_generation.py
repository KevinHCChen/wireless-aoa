import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def gen_points(num_pts, ndim, r=3):
    points = np.random.uniform(-r,r,size=(num_pts,ndim))
    points = np.array(points)
    return points
 



def gen_basestations(num_bases, ndim, r=4, bs_type="unit"):
    # point on the circle with radius r
    #random point and random angle
    if bs_type=="random":
        t = np.random.uniform(0, 2*np.pi, num_bases)
        v = np.random.uniform(0, 2*np.pi, num_bases)
        angles = np.random.uniform(0,180,size=(num_bases,ndim-1)).tolist()
        if ndim == 2:
            bases = zip(zip(r*np.cos(t), r*np.sin(t)), angles)
        elif ndim == 3:
            bases = zip(zip(r*np.cos(t)*np.sin(v), r*np.sin(t)*np.cos(v), r*np.cos(v)), angles)

    elif bs_type=="unit":
        t  = np.arange(0,2*np.pi, (2*np.pi)/num_bases)
        v  = np.arange(0,2*np.pi, (2*np.pi)/num_bases)
        if ndim == 2:
            base_pts  = zip(r*np.cos(t), r*np.sin(t))
            angles = [[np.degrees(np.sign(y)*np.arccos(x/np.linalg.norm([x,y])))+90.] for x, y in base_pts]
        elif ndim == 3:
            # TODO: currently we are assuming z is always 0, may want to change
            base_pts = zip(r*np.cos(t), r*np.sin(t), np.zeros((t.shape[0])))
            # TOD: currently we are assuming 90 degrees for z angle, may want to change

            angles = [[np.degrees(np.sign(y)*np.arccos(x/np.linalg.norm([x,y])))+90, 90.] for x, y, z in base_pts]

        bases = zip(base_pts, angles)

    elif bs_type=="colinear":
        bases = [((4,0), [90.]), ((-4,0), [90.]), ((0,4), [180.])]
    elif bs_type=="structured":
        # bases = [[((4,0), [90.]), ((0,-4), [0.])],[((4,0), [90.]), ((0,4), [0.])]]
        bases = [((4,0), [90.]), ((0,4), [0.]), ((-4,0), [90.]), ((0,4), [0.])]

    return bases



def get_angle(mobile_loc, base_loc, base_theta):
    y = mobile_loc[1] -  base_loc[1]
    x = mobile_loc[0] -  base_loc[0]
    # print "ML: ", mobile_loc
    # print "BL: ", base_loc
    hyp = euclidean(mobile_loc, base_loc)

    if y < 0:
        k = -1
    else:
        k = 1
    angle = np.degrees(k*np.arccos(x/hyp)) % 360
    return angle+base_theta


#mobile_loc - (x,y,z)
# base_lco - (x,y,z)
# base_angles - (alpha, beta, gamma)
def get3D_angles(mobile_loc, base_loc, base_angles):

    pos_idx = range(3)

    mobile_arr = np.array(mobile_loc)
    base_arr = np.array(base_loc)
    base_angles_arr = np.array(base_angles)
    idxs = [[0,1], [1,2], [0,2]]

    return [get_angle(mobile_arr[list(idx)], base_arr[list(idx)], base_angles_arr[idx[0]]) for idx in idxs]


def get_mobile_angles(bases, mobiles, ndim):
    if ndim == 2:
        mobile_angles = []
        # for bases in base_set:
        #   assert mobiles.shape[1]==2, "Mobiles must have 2 columns!"
        #   mobile_angles.append([[get_angle(mobile_loc, base[0], base[1][0]) for base in bases] for mobile_loc in mobiles])
        assert mobiles.shape[1]==2, "Mobiles must have 2 columns!"
        mobile_angles.append([[get_angle(mobile_loc, base[0], base[1][0]) for base in bases] for mobile_loc in mobiles])
        #t1
        angles_output = np.vstack(mobile_angles)
        #t2
        # angles_output = np.hstack(mobile_angles)
        # t3
        # angles_output = mobile_angles
        # print "AO: ", angles_output.shape
        # assert False
    elif ndim == 3:
        print "*****HERE******"
        print "Mobiels: ", mobiles
        print "BASES: ", bases 
        mobile_angles = [[get3D_angles(mobile_loc, base[0], base[1]) for base in bases] for mobile_loc in mobiles]

        mobile_angles = np.array(mobile_angles)
        angles_output = mobile_angles.reshape(mobile_angles.shape[0], -1)

    return angles_output


def generate_data(num_pts, num_stations, ndim, pts_r=3, bs_r=4, bs_type="random"):
    mobiles = gen_points(num_pts, ndim, r=pts_r)
    bases = gen_basestations(num_stations, ndim, r=bs_r, bs_type=bs_type)
    angles = get_mobile_angles(bases, mobiles, ndim)
    return mobiles, bases, angles

def add_noise(data,col_idxs=[-1], noise_type="gaussian", noise_params={'mean': 0, 'std':1}):

    if noise_type == "gaussian":
        assert data.shape[1] >= len(col_idxs), "Bad Noise shape selection!"
        gauss_noise = np.random.normal(loc=noise_params['mean'],\
                                       scale=noise_params['std'],\
                                       size=(data.shape[0],len(col_idxs)))
        data[:,col_idxs] += gauss_noise
    return data



# TODO: maaaaaayyyyyyyybbbbeeeee some assert tests, but everything **looks** great
def test_datagen():
    bs_type = "random"
    ndims = [2,3]
    num_pts = 4
    num_stations = 5
    for ndim in ndims:
        mobiles = gen_points(num_pts, ndim, r=3)
        bases = gen_basestations(num_stations, ndim, bs_type=bs_type)
        angles = get_mobile_angles(bases, mobiles, ndim)
        print "pre noise: ", angles
        angles = add_noise(angles)

        print "************* NDIM=%d ***************" % (ndim)
        print "Mobiles: ", mobiles
        print "Bases: ", bases
        print "Angles: ", angles

test_datagen()

