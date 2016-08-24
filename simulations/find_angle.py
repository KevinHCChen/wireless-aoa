import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import itertools

def to_coef(m, p):
    #conversion from point/slope form of line to standard form. Do on paper if confused.
    const = p[1]-m*p[0]
    #return the left side coefficients and the right side constant. these represent the equation.
    return ([-m, 1], const)


def get_angle(mobile_loc, base_loc, base_theta):
    y = mobile_loc[1] -  base_loc[1]
    x = mobile_loc[0] -  base_loc[0]
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

def test3D_angles():
    num_pts = 100
    t = np.linspace(0,2*np.pi, num_pts)
    #y = np.linspace(-4,4,100)
    #z = np.zeroes(100)
    r = 4

    mobile_pts  = zip(r*np.cos(t), r*np.sin(t), np.ones(num_pts))
    #mobile_pts  = zip(np.zeros(num_pts),r*np.cos(t), r*np.sin(t))
    #mobile_pts  = zip(r*np.cos(t), np.zeros(num_pts),r*np.sin(t))
    base_loc = [0,0,0]
    base_angle = [0,0]

    res = [get3D_angles(mobile_loc, base_loc, base_angle) for mobile_loc in mobile_pts]
    res = np.array(res)
    mobile_pts = np.array(mobile_pts)
    plt.scatter(mobile_pts[:,0], mobile_pts[:,1], c=res[:,2])
    plt.axis('equal')
    return res

def get_angles(bases, mobiles):
    assert mobiles.shape[1]==2, "Mobiles must have 2 columns!"
    mobile_angles = [[get_angle(mobile_loc, base[0], base[1]) for base in bases] for mobile_loc in mobiles]
    """
    for mobile_loc in mobiles:
        base_angles = []
        for base in bases:
            base_angles.append(get_angle(mobile_loc, base[0], base[1]))
        mobile_angles.append(base_angles)
    """
    angles_output = np.vstack(mobile_angles)
    return angles_output, mobiles

def plotStations(baseStations, station_len):
    for bs in baseStations:
        #plt.plot(bs[0][0], bs[0][1], marker=(4, 0, bs[1]), markersize=40)
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1]))
        eq = to_coef(slope, cords)
        plotLine(eq, bs[0])

def plotLine(eq, center):
    x1 = center[0] + .2
    x2 = center[0] - .2
    y1 =  (eq[1] - eq[0][0]*x1)/(eq[0][1])
    y2 =  (eq[1] - eq[0][0]*x2)/(eq[0][1])
    plt.plot([x1,x2], [y1,y2], '-', linewidth=10., markersize=12)



