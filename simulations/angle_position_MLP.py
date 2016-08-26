import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from matplotlib import colors
import sys
import itertools
from sklearn.cross_validation import LeavePOut, train_test_split
from sklearn.svm import SVC as SVM
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import KNeighborsRegressor as KNNR

from MLP import MLPClassifier
from MLP import MLPRegressor
from scipy.stats import norm
import math
from Convert2Complex import *
from find_angle import *

plt.ion()

_colors = ['b','g','r','m','c','k']



def convertPt(x,ang, cords):
    return (cords[0] - x)/np.cos(ang)


def convertToSpherical(X, Y,bases):
    return np.array([[convertPt(y[0], X[i,0], bases[0][0]),convertPt(y[0],X[i,1], bases[1][0])] for i,y in enumerate(Y)])
    #return np.array([np.abs(y[0]/np.cos(X[i,0])) for i,y in enumerate(Y)])

def generateData(num_bases, num_pts=20000, r=5):

    #t = np.random.uniform(0,2*np.pi,num_bases)
    t  = np.arange(0,2*np.pi, (2*np.pi)/num_bases)
    t = t+(np.pi/2.)

    # point on the circle with radius r
    #random point and random angle
    #angles = np.random.randint(0,180,num_bases)
    #bases = zip(zip(r*np.cos(t), r*np.sin(t)), angles)
    base_pts  = zip(r*np.cos(t), r*np.sin(t))

    # random point with parallel angle
    base_angles = [np.degrees(np.sign(y)*np.arccos(x/np.linalg.norm([x,y])))+90 for x, y in base_pts]
    bases = zip(base_pts, base_angles)

    bases = [((4,0), 90), ((0,4), 0), ((0,-4), 0)]
    #bases = [((4,0), 90), ((0,4), 0)]
    #bases = [((-4,0), 90), ((0,4), 0)]
    #bases = [((-4,0), 90), ((0,-4), 0)]
    #bases = [((4,0), 90), ((0,-4), 0)]

    #thetas = np.random.uniform(0,2*np.pi, num_pts)
    #points = [[r*np.cos(thetas[i]), r*np.sin(thetas[i])] for i, r in enumerate(np.random.uniform(0,r, num_pts))]

    points = np.random.uniform(-3,3,size=(num_pts,2))
    points = np.array(points)

    #x = np.linspace(-3,3, int(np.sqrt(num_pts)))
    #points = np.array(np.meshgrid(x, x))
    #points = points.reshape(2,int(np.sqrt(num_pts))**2).T

    X, Y = get_angles(bases,points)
    return X,Y, bases

# expand and only include sets of linearly independent base stations
def expandFeatures(bases,X,Y, eps=5, n_stations=3):
    indep = [base[1]%181 for base in bases]
    Xmat = []
    for j in range(len(bases)):
        dist = np.abs(indep[j] - indep)
        res = np.random.choice(np.where(dist > eps)[0], n_stations, replace=False)
        Xmat.append(X[:,res])
    return np.hstack(Xmat)


def generateRandomData(bases, num_pts=20000):
    points = np.random.uniform(-3,3,size=(num_pts,2))
    points = np.array(points)
    X, Y = get_angles(bases,points)
    return X,Y, bases







########################################################################################
#### Classifier Initialization
########################################################################################

hl_sizes = [(100,), (200,), (500,),(1000,), (100,50), (200,50), (500,50) ]
            #(100,100), (200,100), (500,100),
            #(200,100,20), (500,100,20), (1000,100,20)]


regressors = []
#regressors.append( KNNR(n_neighbors=3))
#regressors.append(SVR(kernel='linear', C=1e3, gamma=0.1))
#regressors.append(SVR(kernel='poly', C=1e3, degree=2))
#regressors.append(DecisionTreeRegressor(max_depth=5))
#regressors.append(RandomForestRegressor(n_estimators=4, max_depth=5 ))
regressors.append( MLPRegressor(hidden_layer_sizes=(500,50), activation='relu', verbose=True,
                                algorithm='adam', alpha=0.000, tol=-float('inf'), early_stopping=False, max_iter=300))

# increasing network sizes:
#for hl_size in hl_sizes:
#    regressors.append( MLPRegressor(hidden_layer_sizes=hl_size, activation='relu', verbose=False,
#                                algorithm='adam', alpha=0.000, tol=1e-8, early_stopping=True))


########################################################################################
#### Test models
########################################################################################


# expand data (corresponds to nonlinear kernels)
#X = np.hstack([X**p for p in range(1,4)])


#plt.figure(1); plt.clf();
base_range = (2,3)
results = []
for i, regressor in enumerate(regressors):
    for num_bases in range(*base_range):
        print "Generating for %d Stations" % (num_bases+1)
        # X = AOA for n base stations
        # Y = labels (x,y) position of mobile nodes
        # bases = base stations position and angle facing
        X, Y, bases = generateData(num_bases+1, num_pts=8000)
        #assert False
        #X = expandFeatures(bases, X, Y, n_stations=3)
        #Y = convertToSpherical(X, Y, bases)
        X = X/360.
        trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.1)
        print "Training and Testing - MLP"
        results.append(regressor.fit(trainX, trainY).score(testX,testY))
        print results[-1]

        predY = regressor.predict(testX)
        error  = np.linalg.norm(predY - testY, axis=1)
        plt.figure()#;plt.clf()
        plt.subplot(2,1,1)
        #plt.scatter(testY[:,0], testY[:,1], c=error)
        plt.scatter(testY[:,0], testY[:,1], c=error)
        plotStations(bases, 2)
        plt.colorbar()
        plt.ylim([-5,5])
        plt.xlim([-5,5])
        plt.title("Num Stations: %d" % (len(bases)))
        plt.show()


        #plt.figure()#;plt.clf()
        plt.subplot(2,1,2)
        plt.scatter(predY[:,0], predY[:,1], c=error)
        plotStations(bases, 2)
        plt.colorbar()
        plt.ylim([-5,5])
        plt.xlim([-5,5])
        plt.title("Num Stations: %d" % (len(bases)))
        plt.show()

        """
        X,Y, bases = generateRandomData(bases, 500)
        X = X/360.
        predY = regressor.predict(X)
        error  = np.linalg.norm(predY - Y, axis=1)
        plt.figure()#;plt.clf()
        #plt.subplot(3,1,3)
        #plt.scatter(testY[:,0], testY[:,1], c=error)
        plt.scatter(Y[:,0], Y[:,1], c=error)
        plotStations(bases, 2)
        plt.colorbar()
        plt.ylim([-5,5])
        plt.xlim([-5,5])
        plt.title("Num Stations: %d" % (num_bases+1))
        plt.show()
        """

# increasing # of units
if False:
    plt.figure()
    plt.plot([np.prod(hl) for hl in hl_sizes], results, '-o')
    plt.xlabel('# Hidden Units')
    plt.ylabel('Prediction Accuracy (MSE)')
    plt.title('Accuracy with Increasing Hidden Units')


