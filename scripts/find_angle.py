import numpy as np
import numpy.linalg as lg
from scipy.spatial.distance import euclidean

def to_coef(m, p):
    #conversion from point/slope form of line to standard form. Do on paper if confused.
    const = p[1]-m*p[0]
    #return the left side coefficients and the right side constant. these represent the equation.
    return ([-m, 1], const)

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
    opp = euclidean(mobile_loc,perp_loc)
    #find the desired angle using arcsin on the known lengths
    return np.degrees(np.arcsin(opp/hyp))
bases = [[[4,4],-45],[[4,-4],20], [[-4,-4],-45], [[-4,4],180]]
mobiles = np.random.uniform(-3,3,(10000,2))
mobile_angles = []
for mobile_loc in mobiles:
    base_angles = []
    for base in bases:
        base_angles.append(get_angle(mobile_loc, base[0], base[1]))
    mobile_angles.append(base_angles)
angles_output = np.vstack(mobile_angles)
with open("mobile_angles.npy", mode="w+b") as out:
    np.save(out, angles_output)
with open("rand_mobiles.npy", mode="w+b") as out:
    np.save(out, mobiles)
