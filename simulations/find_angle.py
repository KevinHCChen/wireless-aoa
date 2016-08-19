import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def to_coef(m, p):
    #conversion from point/slope form of line to standard form. Do on paper if confused.
    const = p[1]-m*p[0]
    #return the left side coefficients and the right side constant. these represent the equation.
    return ([-m, 1], const)

"""
def get_angle(mobile_loc, base_loc, base_theta):
    #get slope of line with angle of base_theta
    m = np.tan(np.radians(base_theta))
    #Slope of line perpendicular to line created by
    mPerp = -1/m
    #get the perpendicular line created by the mobile location and the perpendicular slope in standard form
    mobile_eq = to_coef(mPerp,mobile_loc)
    #get the line created by the base location and the angle's slope in standard form
    base_eq = to_coef(m, base_loc)
    #solve for the intersection of the two lines using a system of linear equations. This returns your 3rd point to make a triangle.
    perp_loc = lg.solve([mobile_eq[0],base_eq[0]], [mobile_eq[1],base_eq[1]])
    #find the lengths of two sides of the triangle.
    hyp = euclidean(mobile_loc,base_loc)
    #opp = euclidean(mobile_loc,perp_loc)
    adj = euclidean(base_loc,perp_loc)

    #find the desired angle using arcsin on the known lengths
    #return np.degrees(np.arcsin(opp/hyp))
    return np.degrees(np.arccos(adj/hyp))
"""

def get_angle(mobile_loc, base_loc, base_theta):
    y = base_loc[1] - mobile_loc[1]
    x =  base_loc[0] - mobile_loc[0]
    hyp = euclidean(mobile_loc, base_loc)

    angle = np.degrees(np.sign(y)*np.arccos(x/hyp)) % 360
    return angle+base_theta



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



