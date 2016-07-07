import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from matplotlib import colors
import sys
import itertools
import re


plt.ion()

######################################################################################################
# Parameters
######################################################################################################

RUNS = [1]


ANG = np.array(range(80,130,10))

# 96 inches (indoor)
# 28.5 feet (outdoor)


##### for 0620 outdoor data
# ANG = np.array(range(60,160,10))
# RUNS = [str(x[0])+'-'+str(x[1]) for x in itertools.product(ANG,range(1,6))]

#### this is for plotting 6-22-2016.MOVEMENT
# ANG = np.array([80])
# RUNS = [str(x[0])+'.'+str(x[1])+'-'+str(x[2]) for x in itertools.product(ANG,[1,2,3], [1])]

##### this is for plotting data/6-22-2016.INDOOR
# RUNS = [str(x[0])+'-'+str(x[1])+'-'+str(x[2]) for x in itertools.product(ANG,range(1,6), [2])]

#### this is for plotting 6-22-2016.MOVEMENT
# line
# RUNS = [str(x[0])+'.'+str(x[1])+'.'+str(x[2]) for x in itertools.product(ANG,[1], range(0,6))]
# stutter
RUNS = [str(x[0])+'.'+str(x[1])+'.'+str(x[2]) for x in itertools.product(ANG,[2], range(0,6))]

#### for 0627
# ANG = np.array(range(80,125,5))
# RUNS = [str(x[0])+'.'+str(x[1]) for x in itertools.product(ANG, range(1,3))]
# ANG = np.array([80])
# RUNS = [str(x[0])+'.'+str(x[1]) for x in itertools.product(ANG, range(1,16))]

# for 0630
ANG = np.array(range(90,130,5))
RUNS = [str(x[0])+'.'+str(x[1])+'-'+str(x[2]) for x in itertools.product([1],ANG, range(0,11))]

# DataFolder = '../testdata/'
# DataFolder = '../data/0621-indoor/'
# DataFolder = '../data/0620-outdoor/'
# DataFolder = '../data/6-22-2016.MOVEMENT/'
# DataFolder = '../data/6-22-2016.INDOOR/'
DataFolder = '../data/6-23-2016.MOVEMENT/STUTTER/'
# DataFolder = '../data/6-23-2016.MOVEMENT/LINE/'
# DataFolder = '../data/0627-stationary-15cm/'
DataFolder = '../data/06-30-StutterAlot/'


if len(sys.argv) > 1:
    RUNS = sys.argv[1:]



N = -1
burn = 200
RX_sequence = [1,3,4,2]     # RX ordering
rx_num = len(RX_sequence)
method = 'Music' # Correlation or Music



LOG_ANG_DIFF = False

# distance_between_rx = -0.088
distance_between_rx = -0.10
distance_between_txrx = 2.44
frequency = 916e6
speedoflight = 3e8
wavelength = speedoflight/frequency


######################################################################################################
# Utility functions
######################################################################################################


_colors = ['b','g','r','m','c','k','lime','orange','brown','gray','tan']+[x for x in colors.cnames]

_colors = [ [c]*11 for c in _colors ]
_colors = [item for sublist in _colors for item in sublist]

def db(power):

    return 10*np.log10(power)
def steeringVector(theta, d_rx=distance_between_rx, d_txrx=distance_between_txrx):
    rx_loc = np.vstack([[rx*d_rx,0] for rx in range(rx_num)])
    rx_loc -= rx_loc.mean(axis=0)
    tx_loc = np.array([d_txrx*np.cos(theta), d_txrx*np.sin(theta)])

    ant_dist = np.array([ np.linalg.norm(tx_loc-x ) for x in rx_loc])
    ant_dist -= ant_dist[0]
    phase_diff = ant_dist*2*np.pi*frequency/speedoflight

    return angle2c( phase_diff )

    # phase_diff = -d_rx*np.cos(theta)*2*np.pi*frequency/speedoflight
    # return angle2c(np.array([rx*phase_diff for rx in range(rx_num)]))

def genSteeringVectors():

    return np.vstack( [ steeringVector(theta) for theta in np.linspace(0,np.pi,180) ] ).T
def angle2c(theta):

    return 1j*np.sin(theta)+np.cos(theta)
def detectSignal(signal, burn_initial_samples=200):
    thres = np.max(np.abs(signal[burn_initial_samples:burn_initial_samples*2]))*5
    t = np.abs(signal) > thres 
    t = binary_dilation(binary_erosion(binary_dilation(t,iterations=10),iterations=25),iterations=100)
    return np.where(t!=0)[0]
def angleDiff(signal1, signal2):
    return np.mod(np.angle(signal1)-np.angle(signal2)+np.pi, 2*np.pi)-np.pi
def readSamples(filename, size = -1):
    dat = np.fromfile(open(filename), dtype=scipy.float32)
    if size == -1:
        size = len(dat)
    dat = dat[:size*2]
    dat_complex = dat[0::2] + dat[1::2]*1j
    return dat_complex
def writeSamples(filename, samples):
    raw_seq = np.vstack([np.real(samples),np.imag(samples)]).T.ravel()
    raw_seq.astype(scipy.float32).tofile(open(filename,'w'))
def plotComplex(samples):
    plt.plot(np.real(samples),np.imag(samples),'.')
def trim_samples(raw_samples, ref):
    trimmed_samples_id = np.array(list( set.union(*[ set(detectSignal(x)) for x in raw_samples])))
    trimmed_samples_id = np.sort(trimmed_samples_id)
    trimmed_samples = [ x[trimmed_samples_id] for x in raw_samples ]

    match = np.abs(np.convolve(np.conjugate(ref[::-1]), trimmed_samples[0],mode='valid'))
    # This is a hack --- find 1 peak in the first half and another peak in the second half
    peaks = np.array([np.argmax( match[:match.size/2]),match.size/2+np.argmax( match[match.size/2:])])
    mask = np.zeros(match.size)
    mask[peaks] = 1
    ref_expanded = np.convolve(ref, mask)
    trimmed_samples = [ x*np.conjugate(ref_expanded) for x in trimmed_samples]
    
    if(ref_expanded[ref_expanded.size/2] != 0):
        print 'The number of samples after trimming does not seem right'
        assert(0)

    return trimmed_samples
def calibration(trimmed_samples):
    # This assumes the first half of samples are coming from reference Tx

    # separate calibration signal
    calibration_samples = [ x[:x.size/2] for x in trimmed_samples ]
    samples = [ x[x.size/2:] for x in trimmed_samples ]

    calib_ang_diff = [angleDiff( calibration_samples[i+1], calibration_samples[0] ) for i in range(rx_num-1) ]
    calib_ang_diff_avg = [0]+[ np.median(x[x.size/4:x.size*3/4]) for x in calib_ang_diff]

    synced_calibration_samples = [ x*angle2c(-y) for x,y in zip(calibration_samples,calib_ang_diff_avg)]
    synced_samples = [ x*angle2c(-y) for x,y in zip(samples,calib_ang_diff_avg)]

    return synced_samples, synced_calibration_samples
def angular_diff(samples):

    return [angleDiff( samples[i+1], samples[i] ) for i in range(len(samples)-1) ]
######################################################################################################
# Plot received signals
######################################################################################################


plt.figure(figsize=(8,4));
labels = []
ALLsamples = []
for run_iterator, run in enumerate(RUNS):

    trimmed_samples = [ readSamples(DataFolder+'trimmed_rx'+str(rx)+'_'+str(run)+'.dat', N) for rx in RX_sequence ] 
    synced_samples, synced_calibration_samples =  calibration(trimmed_samples)
    steeringvectors = genSteeringVectors()
    aggregated_samples = np.vstack(synced_samples)
    ALLsamples.append(aggregated_samples)

    if method == 'Correlation':

        # Correlation approach to determine power coming from each angle    
        angular_power = np.median(np.abs(np.dot( np.conjugate(steeringvectors).T, aggregated_samples)), axis=1)

    elif method == 'Music':

        # the MUSIC approach
        covariance = np.dot(aggregated_samples, np.conjugate(aggregated_samples).T)/aggregated_samples.shape[1]
        eigvalue, eigvector = np.linalg.eig(covariance)

        noisespace = eigvector[:,np.argsort(eigvalue)[:-1]]
        noisepower = np.linalg.norm( np.dot( np.conjugate(noisespace).T, steeringvectors), axis=0)
        angular_power = 1./noisepower


    # Normalize angular power
    # angular_power = angular_power/np.max(angular_power)
    
    ######################################################################################################
    # Plot the angular power
    ######################################################################################################
    angular_power_pt = np.vstack([power*np.array([np.cos(theta), np.sin(theta)])  for power,theta in zip(angular_power, np.linspace(0,np.pi,180)) ])
    plt.plot( angular_power_pt[:,0], angular_power_pt[:,1], _colors[run_iterator],  alpha=0.7)
    theta = np.linspace(0,np.pi,180)[np.argmax(angular_power)]
    pmax = np.max(angular_power)
    plt.plot([0,pmax*np.cos(theta)], [0, pmax*np.sin(theta)], _colors[run_iterator],label=str(run), alpha=0.7)

    angle = int(re.split('-|\.', run)[1])
    runid = int(re.split('-|\.', run)[1])

    if LOG_ANG_DIFF:
        ang_diff = [angleDiff( aggregated_samples[i+1], aggregated_samples[i] ) for i in range(rx_num-1) ]
        median_ang_diff = [np.median(x) for x in ang_diff]
        with open(DataFolder+'ANGDIFF.txt','a') as f:
            f.write(str([angle, runid]+median_ang_diff)[1:-1])
            f.write('\n')

    labels.append([angle, np.linspace(0,np.pi,180)[np.argmax(angular_power)]/np.pi*180 ])

_labels = np.vstack(labels)
error = _labels[:,0]- _labels[:,1]
error_traditional = np.abs(error)
print error_traditional.mean()

######################################################################################################
# Plot auxiliary stuff
######################################################################################################

_,pmax1 = plt.ylim()
pmax2,pmax3 = plt.xlim()
pmax = max(pmax1,-pmax2, pmax3)
plt.ylim([0,pmax])
plt.xlim([-pmax,pmax])

for power in pmax*np.linspace(0.25, 1, 4):
    angular_power_ref = np.vstack([power*np.array([np.cos(theta), np.sin(theta)])  for theta in np.linspace(0,np.pi,180) ])
    plt.plot(angular_power_ref[:,0], angular_power_ref[:,1], 'k-.', alpha=0.5)

for theta in np.linspace(0, np.pi, 19):
    plt.plot([0,pmax*np.cos(theta)], [0, pmax*np.sin(theta)], 'k-.', alpha=1)
plt.legend()


for theta in (ANG/360.)*(2.*np.pi):
    plt.plot([pmax*np.cos(theta)], [pmax*np.sin(theta)], 'ko', alpha=1)
plt.legend()

# plt.savefig('/Users/kevin/Downloads/aoa.pdf')


if len(RUNS)==1:
    plt.figure(figsize=(8,4));
    plt.title('signal magnitude ('+str(run)+')')
    plt.plot(np.abs(trimmed_samples[0]))

    raw_input()


