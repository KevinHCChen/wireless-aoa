import numpy as np
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import matplotlib.pyplot as plt

import logging
import sys

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

def detectSignal(signal, burn_initial_samples=200, thres = 5e-3):
    ''' Returns indexes of samples that are above the threshold (in terms of magnitude)'''
    t = np.abs(signal) > thres 
    t = binary_dilation(t,iterations=200)
    return np.where(t!=0)[0]

def readSamples(filename, maxsize = -1):
    ''' Read samples from file
        The file format is [n0.re, n0.im, n1.re, n1.im, ... ]
    '''
    print 'reading file: {0}'.format(filename)
    dat = np.fromfile(open(filename), dtype=scipy.float32)
    if maxsize == -1:
        maxsize = len(dat)
    dat = dat[:maxsize]
    dat_complex = dat[0::2] + dat[1::2]*1j
    return dat_complex

def writeSamples(filename, samples):
    ''' Write samples to file
        The file format is [n0.re, n0.im, n1.re, n1.im, ... ]
    '''    
    print 'writing file: {0}'.format(filename)
    raw_seq = np.vstack([np.real(samples),np.imag(samples)]).T.ravel()
    raw_seq.astype(scipy.float32).tofile(open(filename,'w'))

def trim_samples(raw_samples, burst_size = 1000, number_of_bursts = 3):

    # Find samples with larger magnitude
    trimmed_samples_id = detectSignal( raw_samples[0] )
    
    if( len(trimmed_samples_id) < number_of_bursts*burst_size ):
        logging.error('Number of samples over the signal detection threshold = {}, Expected at least {}'.format(
            len(trimmed_samples_id), number_of_bursts*burst_size ))
        sys.exit(1)

    # Extract samples with larger magnitude
    trimmed_samples = [ x[trimmed_samples_id] for x in raw_samples ]


    # Convolve with a square function to find the exact location of the bursts
    match = np.convolve(np.ones(burst_size), np.abs( trimmed_samples[0] ),mode='valid')
    peaks = find_peaks(match, thres=0, min_dist=burst_size)

    if( len(peaks) != number_of_bursts):
        logging.error('Found {} bursts\n, Expected {}'.format( len(peaks), number_of_bursts ))
        sys.exit(1)

    # find the indexes of the samples we want to keep
    mask = np.zeros(match.size); mask[peaks] = 1
    idx = np.convolve(np.ones(burst_size), mask)

    if(sum( idx!=0 ) != burst_size*number_of_bursts):
        logging.error('Number of samples after trimming = {}\n, Expected {}'.format( 
            sum( idx!=0 ), burst_size*number_of_bursts ))
        sys.exit(1)

    return [ x[idx!=0] for x in trimmed_samples]

######################################################################################################
# Plot received signals
######################################################################################################

import argparse

parser = argparse.ArgumentParser(description='Extract bursts of signals from raw sample files')
parser.add_argument('--run', type=str, default='0', help='run identifier')
parser.add_argument('--bursts', type=int, default=2, help='number of bursts in the signal')
parser.add_argument('--burst_size', type=int, default=1000, help='samples per burst')
parser.add_argument('--rx', type=int, default=4, help='number of antennas')
parser.add_argument('--path', type=str, default='/root/aoa/data/', help='path to the data folder')

args = parser.parse_args()

# Read samples from raw data files
raw_samples = [ readSamples(args.path+'rx'+str(rx+1)+'_'+str(args.run)+'.dat') for rx in range(args.rx) ] 

# Extract the parts that correspond to the bursts
trimmed_samples = trim_samples(raw_samples, burst_size = args.burst_size, number_of_bursts = args.bursts)

for rx in range(args.rx):
    writeSamples(args.path+'/trimmed_rx'+str(rx+1)+'_'+str(args.run)+'.dat', trimmed_samples[rx])
    

