import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from mpl_toolkits.mplot3d import Axes3D

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

def build_3D_data(bases, mobile_locs):
    mobile_angles = [[get3D_angles(mobile_loc, base[0], base[1]) for base in bases] for mobile_loc in mobile_locs]

    mobile_angles = np.array(mobile_angles)
    angles_output = mobile_angles.reshape(mobile_angles.shape[0], -1)
    return angles_output, mobile_locs

def test3D_angles():
    num_pts = 30
    t = np.linspace(0,2*np.pi, num_pts)
    #y = np.linspace(-4,4,100)
    #z = np.zeroes(100)
    r = 4

    u = np.linspace(0,2*np.pi, num_pts)
    v = np.linspace(0,2*np.pi, num_pts)
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)

    mobile_pts = [(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)) for u,v in itertools.product(u,v)]

    base_loc = [0,0,0]
    base_angle = [0,0]

    res = [get3D_angles(mobile_loc, base_loc, base_angle) for mobile_loc in mobile_pts]
    res = np.array(res)
    mobile_pts = np.array(mobile_pts)
    #plt.scatter(mobile_pts[:,0], mobile_pts[:,1], c=res[:,0])
    #plt.axis('equal')

    plt3d(mobile_pts[:,0], mobile_pts[:,1], mobile_pts[:,2], res[:,0])
    #return fig, ax
    return ()



def plt3d(xs,ys,zs,c):
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=c)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    #plt.show()
    #return fig, ax

def animate(i):
        ax.view_init(elev=10., azim=i)
        return ()

def get_angles(bases, mobiles):
    assert mobiles.shape[1]==2, "Mobiles must have 2 columns!"
    mobile_angles = [[get_angle(mobile_loc, base[0], base[1]) for base in bases] for mobile_loc in mobiles]
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



# test3D_angles()
