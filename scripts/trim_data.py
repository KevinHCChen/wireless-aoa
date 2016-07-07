import numpy as np
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion




def find_peaks(y, thres=0.0, min_dist=1):
    '''Peak detection routine.

    Finds the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks.

    Parameters
    ----------
    y : ndarray
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the indexes of the peaks that were detected
    '''
    thres *= np.max(y) - np.min(y)

    # find the peaks by using the first order difference
    dy = np.diff(y)
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak-min_dist), peak+min_dist+1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def db(power):

    return 10*np.log10(power)
def steeringVector(theta):
    phase_diff = distance_between_rx*np.cos(theta)*2*np.pi*frequency/speedoflight
    return angle2c(np.array([rx*phase_diff for rx in range(rx_num)]))
def genSteeringVectors():

    return np.vstack( [ steeringVector(theta) for theta in np.linspace(0,np.pi,180) ] ).T
def angle2c(theta):

    return 1j*np.sin(theta)+np.cos(theta)
def detectSignal(signal, burn_initial_samples=200):
    # thres = np.max(np.abs(signal))/10
    # thres = np.max(np.abs(signal[burn_initial_samples:burn_initial_samples*3]))*5
    thres  = 5e-3
    t = np.abs(signal) > thres 
    t = binary_dilation(t,iterations=200)
    # t = binary_dilation(binary_erosion(binary_dilation(t,iterations=10),iterations=25),iterations=100)
    return np.where(t!=0)[0]
def angleDiff(signal1, signal2):
    return np.mod(np.angle(signal1)-np.angle(signal2)+np.pi, 2*np.pi)-np.pi
    # return np.minimum(np.angle(signal1)-np.angle(signal2),6.28-(np.angle(signal1)-np.angle(signal2)))
def readSamples(filename, size = -1):
    # print 'reading file: {0}'.format(filename)
    dat = np.fromfile(open(filename), dtype=scipy.float32)
    if size == -1:
        size = len(dat)
    dat = dat[:size*2]
    dat_complex = dat[0::2] + dat[1::2]*1j
    return dat_complex
def writeSamples(filename, samples):
    print 'writing file: {0}'.format(filename)
    raw_seq = np.vstack([np.real(samples),np.imag(samples)]).T.ravel()
    raw_seq.astype(scipy.float32).tofile(open(filename,'w'))
def plotComplex(samples):
    plt.plot(np.real(samples),np.imag(samples),'.')


def feedback(samples, seq_len = 1000):
    shifted_samples = np.zeros(samples.size)
    shifted_samples[seq_len:] = np.abs(samples[:-seq_len])

    return np.maximum(np.abs(samples) - shifted_samples, 0)

def trim_samples(raw_samples, ref):

    ref = np.ones(1000)

    trimmed_samples_id = np.array(list( set.union(*[ set(detectSignal(x)) for x in raw_samples])))
    print "number of samples = {0}".format(len(trimmed_samples_id))
    trimmed_samples_id = np.sort(trimmed_samples_id)
    trimmed_samples = [ x[trimmed_samples_id] for x in raw_samples ]



    match = (np.convolve(ref[::-1], np.abs( feedback(trimmed_samples[0]) ),mode='valid'))
    # match = np.abs(np.convolve(np.conjugate(ref[::-1]), np.abs(trimmed_samples[0]),mode='valid'))
    # match = np.abs(np.convolve(np.conjugate(ref[::-1]), (trimmed_samples[0]),mode='valid'))

    # ref[:100] = 0
    # ref[-100:] = 0

    # This is a hack --- find 1 peak in the first half and another peak in the second half
    # peaks = np.array([np.argmax( match[:match.size/2]),match.size/2+np.argmax( match[match.size/2:])])
    # peaks = np.argsort( match )[-2:]
    # peaks = findpeak(match, np.arange(1, 20))

    peaks = find_peaks(match, thres=0, min_dist=1000)

    print 'peaks: '+str(peaks)

    mask = np.zeros(match.size)
    mask[peaks] = 1
    ref_expanded = np.convolve(ref, mask)
    _trimmed_samples = [ x*np.conjugate(ref_expanded) for x in trimmed_samples]
    # trimmed_samples = [ x/ref_expanded for x in trimmed_samples]
    # trimmed_samples = trimmed_samples[ref_expanded!=0]
    print 'ref_expanded: '+str( sum(ref_expanded!=0)  )

    # if(ref_expanded[ref_expanded.size/2] != 0):
    if(sum( ref_expanded!=0 ) != 2000):
        print 'The number of samples after trimming ({}) does not seem right'.format(sum( ref_expanded!=0 ))
        assert(0)

    # print np.where(ref_expanded!=0)[0]
    return [ x[ref_expanded!=0] for x in _trimmed_samples]

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

import sys 

if len(sys.argv) == 3:
    run_id = sys.argv[1]
    ref_file = sys.argv[2]
else:
    print """
            please provide run id and reference_file

          """

print sys.argv     

aoa_path = '/root/aoa/'
RX = 4

raw_samples = [ readSamples(aoa_path+'data/rx'+str(rx)+'_'+str(run_id)+'.dat', 1e7) for rx in range(1,RX+1) ] 


ref = readSamples(ref_file)
trimmed_samples = trim_samples(raw_samples, ref)

for rx in range(1,RX+1):
    writeSamples(aoa_path+'data/trimmed_rx'+str(rx)+'_'+str(run_id)+'.dat', trimmed_samples[rx-1])
    

