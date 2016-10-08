import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean


def gen_points_random(num_pts, ndim, r=3):
    points = np.random.uniform(-r,r,size=(num_pts,ndim))
    points = np.array(points)
    return points

# Returns a grid of points with a square or cube root of the num_pts requests
def gen_points_grid(num_pts, ndim, r=3):
    assert ndim <= 3, "gen_points_grid in data_generation has not been implemented yet for %d dimensions" % (ndim)
    x = np.linspace(-r,r, int(np.power(num_pts, 1./ndim)))
    points = np.array(np.meshgrid(*([x]*ndim)))
    points = points.reshape(ndim,int(np.power(num_pts, 1./ndim))**ndim).T
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
        #bases = [((4,0), [90.]), ((-4,0), [90.]), ((0,4), [180.])]
        bases = [((4,0), [90.]), ((-4,0), [270.]), ((0,4), [180.])]
        # bases = [((-4,0), [90.]), ((0,4), [180.])]
    elif bs_type=="colinear-3D":
        bases = [((4,0,0), [90.,90.]), ((-4,0,0), [90.,90.]), ((0,4,0), [180.,90.])]
    elif bs_type=="structured":
        #bases = [[((4,0), [90.]), ((0,-4), [0.])],[((4,0), [90.]), ((0,4), [0.])]]
        bases = [((4,0), [90.]), ((-4,0), [270.]), ((0,4), [180.])]
        #bases = [((4,0), [90.]), ((0,4), [0.]), ((-4,0), [90.]), ((0,4), [0.])]
    elif bs_type=="structured-3D":
        bases = [((4,0,0), [90.,90.]), ((-4,0,0), [90.,90.]), ((0,4,0), [180.,90.])]

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
        mobile_angles = [[get3D_angles(mobile_loc, base[0], base[1]) for base in bases] for mobile_loc in mobiles]

        mobile_angles = np.array(mobile_angles)
        angles_output = mobile_angles.reshape(mobile_angles.shape[0], -1)

    return angles_output


def generate_data(num_pts, num_stations, ndim, pts_r=3, bs_r=4, bs_type="random", points_type="random"):
    if points_type == "random":
        mobiles = gen_points_random(num_pts, ndim, r=pts_r)
    elif points_type == "grid":
        mobiles = gen_points_grid(num_pts, ndim, r=pts_r)
    else:
        assert False, "This pattern of point generation has not been implemented yet in data_generation"
    bases = gen_basestations(num_stations, ndim, r=bs_r, bs_type=bs_type)
    angles = get_mobile_angles(bases, mobiles, ndim)
    angles %= 360
    angles /= 360.
    angles = np.nan_to_num(angles)
    return mobiles, bases, angles



def replicate_data(data, ndim,  base_idxs):

    if len(base_idxs) == 0:
        return data
    else:
        # using 1 angle for each base station (alpha)
        if ndim == 2:
            rep_data = []
            for idxs in base_idxs:
                print idxs
                rep_data.append(data[:,idxs])
            data = np.hstack(rep_data)
        # using 3 angles for each base station (alpha beta gamma)
        elif ndim == 3:
            rep_data = []
            for idxs in base_idxs:
                col_idxs = []
                for i in idxs:
                    col_idxs += [i*ndim, i*ndim+1, i*ndim+2]
                rep_data.append(data[:,col_idxs])
            data = np.hstack(rep_data)

    return data



# TODO: maaaaaayyyyyyyybbbbeeeee some assert tests, but everything **looks** great
def test_datagen():
    bs_type = "random"
    ndims = [2,3]
    num_pts = 4
    num_stations = 5
    for ndim in ndims:
        mobiles = gen_points_random(num_pts, ndim, r=3)
        bases = gen_basestations(num_stations, ndim, bs_type=bs_type)
        angles = get_mobile_angles(bases, mobiles, ndim)
        print "pre noise: ", angles
        angles = add_noise(angles)

        print "************* NDIM=%d ***************" % (ndim)
        print "Mobiles: ", mobiles
        print "Bases: ", bases
        print "Angles: ", angles

#test_datagen()

