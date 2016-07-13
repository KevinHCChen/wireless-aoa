import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from matplotlib import colors
import sys
import itertools
from sklearn.cross_validation import LeavePOut, train_test_split
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import KNeighborsRegressor as KNNR

from MLP import MLPClassifier
from MLP import MLPRegressor
import math

plt.ion()

########################################################################################
#### Parameters
########################################################################################

# 96 inches (indoor)
# 28.5 feet (outdoor)
rx_num = 4
distance_between_rx = 0.15
frequency = 916e6
speedoflight = 3e8
wavelength = speedoflight/frequency


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
#### Classifier Initialization
########################################################################################

regressors = []
regressors.append( KNNR(n_neighbors=3))
regressors.append( MLPRegressor(hidden_layer_sizes=(200, 50, 50), activation='relu', verbose=False,
    algorithm='adam', alpha=0.000, tol=1e-5, early_stopping=True))



########################################################################################
#### Data Generation
########################################################################################

# random angle  between 80 and 120 degree
Y = np.random.randint(800,1200, size=(1200))/10.

# expected phase difference of received samples
X = np.vstack( [ angular_diff(steeringVector(theta)) for theta in Y/180.*np.pi ] )
X = X + np.random.randn(X.shape[0], X.shape[1])*5e-2


########################################################################################
#### Test models
########################################################################################


# expand data (corresponds to nonlinear kernels)
X = np.hstack([X**p for p in range(1,4)])

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.5)

plt.figure(1); plt.clf();
for i, regressor in enumerate(regressors):
    regressor.fit(trainX, trainY)

    plt.plot(testY, regressor.predict(testX) , _colors[i]+'o', alpha=0.4, label=regressor.__class__.__name__)
    plt.plot(trainY, regressor.predict(trainX) , _colors[i]+'.', alpha=0.4)

    print "{} precision = {:.4f}".format(regressor.__class__.__name__, regressor.score(testX, testY))

plt.plot(testY, testY, 'k-.', alpha=1, label='ground truth')
plt.legend(loc='best')




