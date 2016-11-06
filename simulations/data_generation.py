import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean

speed_of_light = 299792458.0
lambda_val = 0.15
num_antennas_per_bs = 4
freq = .1*1e9
period = 1./(freq)

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


def get_phaseoffset(mobile_loc, base_loc, base_theta):

    arrival_times = [euclidean(mobile_loc, ((base_loc[0] + (i*lambda_val)), base_loc[1]))/speed_of_light for i in range(num_antennas_per_bs)]

    arrival_times = np.array(arrival_times) #% period
    phase_offsets = arrival_times - arrival_times[0]
    phase_offsets = phase_offsets[1:]

    return phase_offsets




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
    elif bs_type=="outdoor":
        #bases = [((4,0), [90.]), ((-4,0), [90.]), ((0,4), [180.])]
        bases = [((-3.58,0), [90.]),((0, 2.726), [180.]), ((3.58,0), [270.])]
        # bases = [((-4,0), [90.]), ((0,4), [180.])]
    elif bs_type=="colinear-3D":
        bases = [((4,0,0), [90.,90.]), ((-4,0,0), [90.,90.]), ((0,4,0), [180.,90.])]
    elif bs_type=="structured":
        #bases = [[((4,0), [90.]), ((0,-4), [0.])],[((4,0), [90.]), ((0,4), [0.])]]
        bases = [((4,0), [90.]), ((-4,0), [270.]), ((0,4), [180.])]
        #bases = [((4,0), [90.]), ((0,4), [0.]), ((-4,0), [90.]), ((0,4), [0.])]
    elif bs_type=="structured-3D":
        bases = [((4,0,0), [90.,90.]), ((-4,0,0), [90.,90.]), ((0,4,0), [180.,90.])]
    elif bs_type=="faraway":
        #bases = [((-3,18), [180.]), ((3,18), [180.])]
        #bases = [((-3,18), [0.]), ((3,18), [0.])]
        bases = [((-18,0), [90.]), ((18,0), [90.])]
        bases = [((-18,-18), [0.]), ((18,18), [0.])]
        bases = [((-18,18), [0.]), ((18,18), [0.])]
        bases = [((-18,-18), [0.]), ((18,-18), [0.])]
    elif bs_type=="faraway_but_closer":
        bases = [((-3,4), [0.]), ((3,4), [0.])]

    return bases


def generate_from_real_data(parsed_data):
    base_stations = [((-3.58,0), [90.]),((0, 2.726), [180.]), ((3.58,0), [270.])]
    virt_x=np.tile(np.linspace(-2.5,2.5, 6), 5)
    virt_y=np.repeat(np.linspace(-1.6,1.6, 5), 6)

    mobiles = np.vstack((virt_x, virt_y)).T
    mobiles = np.tile(mobiles,(parsed_data.shape[1],1))
    # true_angles = get_mobile_angles(base_stations, mobiles, 2)
    # true_angles %= 360

    angles = []
    for run_number in range(parsed_data.shape[0]):
        # xx= 180. - parsed_data[1,run_number,:,:].T.ravel()
        # print xx.shape
        # print xx
        # assert False
        this_run_angles = np.vstack(([180. - parsed_data[i,run_number,:,:].T.ravel() for i in range(parsed_data.shape[0])]))
        angles.append(this_run_angles.T)

    angles = np.vstack(angles)
    # angles = angles.T
    # print angles
    # print angles.shape
    # print mobiles
    # assert False

    # angles = np.vstack(([np.hstack(([180. - parsed_data[i,j,:,:].T.ravel() for i in range(parsed_data.shape[0])]))\
                                                                           # for j in range(parsed_data.shape[1])]))

    # angles = angles.T


    angles %= 360
    angles /= 360.

    return mobiles, base_stations, angles

    
    # print angles.shape
    # print mobiles.shape
    # assert False, 'Snapp, I cant believe this worked'








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
    theta = np.degrees(np.arccos(np.abs(x)/np.abs(hyp)))
    angle = theta

    if x > 0 and y > 0:
        angle = theta
    if x < 0 and y > 0:
        angle = 180. - theta
    if x < 0 and y < 0:
        angle = 180. + theta
    if x > 0 and y < 0:
        angle = 360. - theta
    # return (angle+base_theta) % 360.
    return angle % 360

    #angle = np.degrees(k*np.arccos(x/hyp)) % 360
    #return angle+base_theta


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

def get_mobile_phases(bases, mobiles, ndim):
    if ndim == 2:
        mobile_angles = []
        assert mobiles.shape[1]==2, "Mobiles must have 2 columns!"
        mobile_angles.append([[get_phaseoffset(mobile_loc, base[0], base[1][0]) for base in bases] for mobile_loc in mobiles])
        angles_output = np.vstack(mobile_angles)
        angles_output = angles_output.reshape(-1,angles_output.shape[1]*angles_output.shape[2])
        angles_output *= 1e10
        print angles_output
        # print angles_output
        # print angles_output.shape
        # assert False
    elif ndim == 3:
        assert False
        # mobile_angles = [[get3D_angles(mobile_loc, base[0], base[1]) for base in bases] for mobile_loc in mobiles]
        # mobile_angles = np.array(mobile_angles)
        # angles_output = mobile_angles.reshape(mobile_angles.shape[0], -1)

    return angles_output


def generate_data(num_pts, num_stations, ndim, pts_r=3, bs_r=4, bs_type="random", points_type="random"):
    if points_type == "random":
        mobiles = gen_points_random(num_pts, ndim, r=pts_r)
    elif points_type == "grid":
        mobiles = gen_points_grid(num_pts, ndim, r=pts_r)
    else:
        assert False, "This pattern of point generation has not been implemented yet in data_generation"
    bases = gen_basestations(num_stations, ndim, r=bs_r, bs_type=bs_type)
    phases = get_mobile_phases(bases, mobiles, ndim)
    angles = get_mobile_angles(bases, mobiles, ndim)
    # angles = np.abs(90. - angles)
    
    angles %= 360
    angles /= 360.
    angles = np.nan_to_num(angles)
    # angles = angles - np.mean(angles,axis=0)

    return mobiles, bases, angles, phases



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
# to cite sources, this comes from...fuck that shit
import math
def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in xrange(0,n+1)]


def test_angle_calculation():
    r = 4.
    bases = [((0.,0.), [0.])]
    circle_points = np.array(PointsInCircum(1,100))
    angles = get_mobile_angles(bases, circle_points, 2)
    print angles
    print "Did it work? [Y]/n"
    print "> "
    return


# test_angle_calculation()

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








