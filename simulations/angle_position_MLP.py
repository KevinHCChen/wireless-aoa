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


def generatePts():
    pass



def generateData(num_bases, num_pts=20000, r=5):

    #t = np.random.uniform(0,2*np.pi,num_bases)
    t  = np.arange(0,2*np.pi, (2*np.pi)/num_bases)
    #t = t+(np.pi/4.)

    # point on the circle with radius r
    #random point and random angle
    #angles = np.random.randint(0,180,num_bases)
    #bases = zip(zip(r*np.cos(t), r*np.sin(t)), angles)
    base_pts  = zip(r*np.cos(t), r*np.sin(t))

    # random point with parallel angle
    base_angles = [np.degrees(np.sign(y)*np.arccos(x/np.linalg.norm([x,y])))+90 for x, y in base_pts]
    bases = zip(base_pts, base_angles)

    #thetas = np.random.uniform(0,2*np.pi, num_pts)
    #points = [[r*np.cos(thetas[i]), r*np.sin(thetas[i])] for i, r in enumerate(np.random.uniform(0,r, num_pts))]

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
#regressors.append( MLPRegressor(hidden_layer_sizes=(500,50), activation='relu', verbose=False,
#                                algorithm='adam', alpha=0.000, tol=1e-8, early_stopping=True))

for hl_size in hl_sizes:
    regressors.append( MLPRegressor(hidden_layer_sizes=hl_size, activation='relu', verbose=False,
                                algorithm='adam', alpha=0.000, tol=1e-8, early_stopping=True))


########################################################################################
#### Test models
########################################################################################


# expand data (corresponds to nonlinear kernels)
#X = np.hstack([X**p for p in range(1,4)])


#plt.figure(1); plt.clf();
base_range = (3,4)
results = []
for i, regressor in enumerate(regressors):
    for num_bases in range(*base_range):
        print "Generating for %d Stations" % (num_bases+1)
        X, Y, bases = generateData(num_bases+1, num_pts=5000)
        X = X/360.
        trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.1)
        print "Training and Testing - MLP"
        results.append(regressor.fit(trainX, trainY).score(testX,testY))
        print results[-1]

    #plt.plot(testY, regressor.predict(testX) , _colors[i]+'o', alpha=0.4, label=regressor.__class__.__name__)
    #plt.plot(trainY, regressor.predict(trainX) , _colors[i]+'.', alpha=0.4)
    #plt.plot(testY, regressor.predict(testX) , 'o', alpha=0.4, label=regressor.__class__.__name__)
    #plt.plot(trainY, regressor.predict(trainX) ,'.', alpha=0.4)

    #print "{} precision = {:.4f}".format(regressor.__class__.__name__, regressor.score(testX, testY))
        predY = regressor.predict(testX)
        error  = np.linalg.norm(predY - testY, axis=1)

        """
        plt.figure()#;plt.clf()
        plt.plot(np.arange(base_range)+1, results, '-o')
        plt.xlabel('Num Base Stations')
        plt.ylabel('1-Layer MLP - Precision')
        #plt.savefig('MLP_IncreasingBaseStations_500')
        plt.show()
        """
        plt.figure()#;plt.clf()
        plt.subplot(2,1,1)
        plt.scatter(testY[:,0], testY[:,1], c=error)
        plotStations(bases, 2)
        plt.colorbar()
        plt.ylim([-5,5])
        plt.xlim([-5,5])
        plt.title("Num Stations: %d" % (num_bases+1))
        plt.show()

        #plt.figure()#;plt.clf()
        plt.subplot(2,1,2)
        plt.scatter(predY[:,0], predY[:,1], c=error)
        plotStations(bases, 2)
        plt.colorbar()
        plt.ylim([-5,5])
        plt.xlim([-5,5])
        plt.title("Num Stations: %d" % (num_bases+1))
        plt.show()

#plt.plot(testY, testY, 'k-.', alpha=1, label='ground truth')
#plt.legend(loc='best')
#plt.show()


plt.figure()
plt.plot([np.prod(hl) for hl in hl_sizes], results, '-o')

#for xy in zip([np.prod(hl) for hl in hl_sizes], ["{0:.2f}".format(r) for r in results]):
#    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('# Hidden Units')
plt.ylabel('Prediction Accuracy (MSE)')
plt.title('Accuracy with Increasing Hidden Units')
