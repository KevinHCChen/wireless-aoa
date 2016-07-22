import numpy as np
import itertools


def genInfo():

    ANG = np.array(range(60,160,10))
    RUNS = [str(x[0])+'-'+str(x[1]) for x in itertools.product(ANG,range(1,6))]

    _colors = ['b','g','r','m','c','k','lime','orange','brown','gray','tan']
    _colors = [ [c]*5 for c in _colors ]
    _colors = [item for sublist in _colors for item in sublist]

    return np.array(ANG), RUNS, _colors    
    
