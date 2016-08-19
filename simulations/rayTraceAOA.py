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
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import KNeighborsRegressor as KNNR

from MLP import MLPClassifier
from MLP import MLPRegressor
from scipy.stats import norm
import math
from Convert2Complex import *

plt.ion()


########################################################################################
#### Utilities
########################################################################################

def angle2c(theta):
    return 1j*np.sin(theta)+np.cos(theta)
def steeringVector(theta):
    phase_diff = distance_between_rx*np.cos(theta)*2*np.pi/wavelength
    return angle2c(np.array([rx*phase_diff for rx in range(rx_num)]))
def angleDiff(signal1, signal2):
    return np.mod(np.angle(signal1)-np.angle(signal2)+np.pi, 2*np.pi)-np.pi    
def angular_diff(samples):
    return [angleDiff( samples[i+1], samples[i] ) for i in range(len(samples)-1) ]
def shift2degree(x):
    return np.arccos(x)/np.pi*180.
def degree2shift(x):
    return (np.cos(x/180.*np.pi))

def identity(x):
    return x

_colors = ['b','g','r','m','c','k']



########################################################################################
#### Load Data
########################################################################################

X, Y = loadData('../data/1800TxTry2/','', 1)
#X = np.load('X.npy')
#Y = np.load('Y.npy')

X = X[:800,:]
Y = Y[:800]
rep_factor = 2
X = np.repeat(X,rep_factor,axis=0)
Y = np.repeat(Y,rep_factor,axis=0)
X = X + norm.rvs(0, 1e-8, size=X.shape)
X = np.angle(X)


X = X[:,1:] - X[:,:-1]
########################################################################################
#### Classifier Initialization
########################################################################################

hl_sizes = [(1000,), (100,50), (200,50), (500,50), (1000,50), \
            (100,100), (200,100), (500,100), (1000,100), \
            (200,100,20), (500,100,20), (1000,100,20)]


regressors = []
#regressors.append( KNNR(n_neighbors=3))
regressors.append(SVR(kernel='linear', C=1e3, gamma=0.1))
regressors.append(SVR(kernel='poly', C=1e3, degree=2))
regressors.append( MLPRegressor(hidden_layer_sizes=(800,50), activation='relu', verbose=False,
                                algorithm='adam', alpha=0.000, tol=1e-8, early_stopping=True))
#for hl_size in hl_sizes:
#    regressors.append( MLPRegressor(hidden_layer_sizes=hl_size, activation='relu', verbose=False,
#                                algorithm='adam', alpha=0.000, tol=1e-8, early_stopping=True))


########################################################################################
#### Test models
########################################################################################


# expand data (corresponds to nonlinear kernels)
#X = np.hstack([X**p for p in range(1,4)])
#Y = Y/360.

test_size=.2

plt.figure(1); plt.clf();
for i, regressor in enumerate(regressors):

    if regressor.__class__.__name__ != "SVR":
        Y = Y/360.
        trainX, testX, trainY, testY = train_test_split(X,Y, test_size=test_size)
        regressor.fit(trainX, trainY)
        plt.plot(testY*360., regressor.predict(testX)*360. , _colors[i]+'o', alpha=0.8, label=regressor.__class__.__name__)
        #plt.plot(trainY*360., regressor.predict(trainX)*360. , _colors[i]+'.', alpha=0.4)
    else:
        trainX, testX, trainY, testY = train_test_split(X,Y, test_size=test_size)
        regressor.fit(trainX, trainY)
        plt.plot(testY, regressor.predict(testX) , _colors[i]+'o', alpha=0.8, label=regressor.__class__.__name__)
        #plt.plot(trainY, regressor.predict(trainX) , _colors[i]+'.', alpha=0.4)

    #plt.plot(testY, regressor.predict(testX) , 'o', alpha=0.4, label=regressor.__class__.__name__)
    #plt.plot(trainY, regressor.predict(trainX) ,'.', alpha=0.4)

    print "{} precision = {:.4f}".format(regressor.__class__.__name__, regressor.score(testX, testY))
    #print "sizes: %s" % (regressor.hidden_layer_sizes,)

plt.plot(testY*360., testY*360., 'k-.', alpha=1, label='ground truth')
plt.legend(loc='best')
plt.xlim([260,370])
plt.ylim([260,370])
plt.xlabel("Ground Truth AOA")
plt.ylabel("Predicted AOA")




