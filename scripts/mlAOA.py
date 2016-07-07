import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from matplotlib import colors
import sys
import itertools
from sklearn.cross_validation import LeavePOut
# from sklearn.svm import LinearSVC as SVM
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN

from MLP import MLPClassifier
from MLP import MLPRegressor
import math


# 96 inches (indoor)
# 28.5 feet (outdoor)


rx_num = 4
distance_between_rx = 0.088
frequency = 916e6
speedoflight = 3e8
wavelength = speedoflight/frequency

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


plt.ion()


DataFolder = '../data/0620-outdoor/'
# DataFolder = '../data/0621-indoor/'


data = np.loadtxt(DataFolder+'ANGDIFF.txt',delimiter=',')

mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation='tanh', algorithm='adam', alpha=0.0001)
mlp_reg = MLPRegressor(hidden_layer_sizes=(200, 50, 50), activation='relu', verbose=True,
    algorithm='adam', alpha=0.000, tol=1e-5, early_stopping=True)

# from sklearn.svm import SVC as SVM
svm = SVM(C=100)


# classifier = svm
classifier = mlp
classifier = mlp_reg





if False:
    X = data[:,-3:]
    Y = data[:,0].astype(int)
    # Y = np.cos(Y/180.*np.pi)
    # Y = degree2shift(Y)
    ID = data[:,1]
    f = identity
else:
    Y = np.random.randint(800,1200, size=(300))/10.
    X = np.vstack( [ angular_diff(steeringVector(theta)) for theta in Y/180.*np.pi ] )
    X = X + np.random.randn(X.shape[0], X.shape[1])*5e-2
    # Y = np.cos(Y/180.*np.pi)
    ID = Y

    f = identity

    # Y = degree2shift(Y)
    # f = shift2degree
    


# Start doing the standard thing

# X = np.hstack([X**p/math.factorial(p) for p in range(1,5)])
X = np.hstack([X**p for p in range(1,4)])

plt.clf()
plt.plot(f(Y), f(Y) ,'o-', alpha=0.3)
plt.plot(f(Y), f(classifier.fit(X,Y).predict(X)) ,'.')

# assert(0)

score = []
labels = []
max_run = 100
for train_index, test_index in LeavePOut(len(X), 1):
    max_run -= 1
    if max_run <1:
        break
    classifier.fit(X[train_index,:],Y[train_index])
    score.append( classifier.score(X[test_index,:],Y[test_index]) )
    # print classifier.score(X[test_index,:],Y[test_index]), Y[test_index], ID[test_index]
    labels.append([classifier.predict(X[test_index,:]), Y[test_index]])
    if score[-1] != 1:
        print Y[test_index], classifier.predict(X[test_index,:]), ID[test_index]

from sklearn.metrics import confusion_matrix as cm
labels = np.hstack(labels)
error = (labels[0]-labels[1])
error_ml = np.abs(error)

# plt.plot(Y, Y ,'o-', alpha=0.3)
plt.plot(f(labels[1]), f(labels[0]),'.')


if False:
    Y = (np.random.randint(300,1200, size=(1500)))/10.
    X = np.vstack( [ angular_diff(steeringVector(theta)) for theta in Y/180.*np.pi ] )
    X = np.hstack([X**p for p in range(1,4)])
    Y = np.cos((180-Y)/180.*np.pi)
    ID = Y    
    classifier.fit(X,Y)

    X = data[:,-3:]
    X = np.hstack([X**p for p in range(1,4)])
    Y = data[:,0].astype(int)
    Y = np.cos(Y/180.*np.pi)
    ID = data[:,1]    

    plt.clf(); plt.plot(f(Y), f(classifier.predict(X)),'.')

    labels = [f(Y), f(classifier.predict(X))]
    error = (labels[0]-labels[1])
    error_ml = np.abs(error)
    print np.mean(error_ml)


