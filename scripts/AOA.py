import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from matplotlib import colors
import sys
import itertools
import re

plt.ion()

import argparse
parser = argparse.ArgumentParser(
    description='plot signal strength for different AOA.')
parser.add_argument('--run', type=str, default=None, help='run identifier')
parser.add_argument('--rx', type=int, default=4, help='number of antennas')
parser.add_argument('--path', type=str,
                    default='../data/testdata/', help='path to the data folder')
parser.add_argument('--method', type=str, default='Music',
                    choices=['Music', 'Correlation', "rMusic"], help='algorithm for AOA detection')
parser.add_argument('--dRxRx', type=float, default=-0.15,
                    help='distance between rx antennas (meters)')
parser.add_argument('--dRxTx', type=float, default=2.44,
                    help='distance between rx and Tx (meters)')
parser.add_argument('--freq', type=float, default=916e6, help='ant frequency')
parser.add_argument('--targets', type=int, default=1,
                    help='how many targets sent signals')
parser.add_argument('--burst_size', type=int, default=1000,
                    help='how many samples do you have in each burst')

args = parser.parse_args()

if args.run == None:
    sys.path.append(args.path)
    from genInfo import genInfo
    ANG, RUNS, _colors = genInfo()
else:
    ANG = np.array([])
    RUNS = args.run.split('+')
    _colors = ['b', 'g', 'r', 'm', 'c', 'k',
               'lime', 'orange', 'brown', 'gray', 'tan']


speedoflight = 3e8


##########################################################################
# Utility functions
##########################################################################

def db(power):
    return 10 * np.log10(power)


def steeringVector(theta, d_rx=args.dRxRx, d_txrx=args.dRxTx, frequency=args.freq):
    rx_loc = np.vstack([[rx * d_rx, 0] for rx in range(args.rx)])
    rx_loc -= rx_loc.mean(axis=0)
    tx_loc = np.array([d_txrx * np.cos(theta), d_txrx * np.sin(theta)])

    ant_dist = np.array([np.linalg.norm(tx_loc - x) for x in rx_loc])
    ant_dist -= ant_dist[0]
    phase_diff = ant_dist * 2 * np.pi * frequency / speedoflight

    return angle2c(phase_diff)


def genSteeringVectors():

    return np.vstack([steeringVector(theta) for theta in np.linspace(0, np.pi, 180)]).T


def angle2c(theta):

    return 1j * np.sin(theta) + np.cos(theta)


def angleDiff(signal1, signal2):
    return np.mod(np.angle(signal1) - np.angle(signal2) + np.pi, 2 * np.pi) - np.pi


def readSamples(filename, size=-1):
    dat = np.fromfile(open(filename, 'rb'), dtype=scipy.float32)
    if size == -1:
        size = len(dat)
    dat = dat[:size * 2]
    dat_complex = dat[0::2] + dat[1::2] * 1j
    return dat_complex


def writeSamples(filename, samples):
    raw_seq = np.vstack([np.real(samples), np.imag(samples)]).T.ravel()
    raw_seq.astype(scipy.float32).tofile(open(filename, 'wb'))


def plotComplex(samples):
    plt.plot(np.real(samples), np.imag(samples), '.')


def calibration(trimmed_samples):
    # This assumes the first half of samples are coming from reference Tx

    # separate calibration signal
    calibration_samples = [x[:x.size / 2] for x in trimmed_samples]
    samples = [x[x.size / 2:] for x in trimmed_samples]

    calib_ang_diff = [angleDiff(
        calibration_samples[i + 1], calibration_samples[0]) for i in range(args.rx - 1)]
    calib_ang_diff_avg = [
        0] + [np.median(x[x.size / 4:x.size * 3 / 4]) for x in calib_ang_diff]

    synced_calibration_samples = [
        x * angle2c(-y) for x, y in zip(calibration_samples, calib_ang_diff_avg)]
    synced_samples = [x * angle2c(-y)
                      for x, y in zip(samples, calib_ang_diff_avg)]

    return synced_samples, synced_calibration_samples


def angular_diff(samples):

    return [angleDiff(samples[i + 1], samples[i]) for i in range(len(samples) - 1)]
##########################################################################
# Plot received signals
##########################################################################


plt.figure(figsize=(8, 4))
labels = []
ALLsamples = []
for run_iterator, run in enumerate(RUNS):

    trimmed_samples = [readSamples(
        args.path + 'trimmed_rx' + str(rx + 1) + '_' + str(run) + '.dat') for rx in range(args.rx)]
    synced_samples, synced_calibration_samples = calibration(trimmed_samples)
    steeringvectors = genSteeringVectors()
    aggregated_samples = np.vstack(synced_samples)
    ALLsamples.append(aggregated_samples)

    if args.method == 'Correlation':

        # Correlation approach to determine power coming from each angle
        angular_power = np.median(
            np.abs(np.dot(np.conjugate(steeringvectors).T, aggregated_samples)), axis=1)

    elif args.method == 'Music':

        # the MUSIC approach
        covariance = np.dot(aggregated_samples, np.conjugate(
            aggregated_samples).T) / aggregated_samples.shape[1]
        eigvalue, eigvector = np.linalg.eig(covariance)

        noisespace = eigvector[:, np.argsort(eigvalue)[:-1]]
        noisepower = np.linalg.norm(
            np.dot(np.conjugate(noisespace).T, steeringvectors), axis=0)
        angular_power = 1. / noisepower

    elif args.method == 'rMusic':
        Fe = args.freq
        L = args.targets
        M = args.burst_size
        x = aggregated_samples
        N = x.shape[0]

        if M == None:
            M = N // 2

        # extract noise subspace
        covariance = np.dot(aggregated_samples, np.conjugate(
            aggregated_samples).T) / aggregated_samples.shape[1]
        U, S, V = np.linalg.svd(covariance)
        eigvalue, eigvector = np.linalg.eig(covariance)
        noisespace = eigvector[:, np.argsort(eigvalue)[:-1]]
        # construct matrix P
        P = np.dot(noisespace,  noisespace.conjugate().T)

        # construct polynomial Q
        Q = 0j * np.zeros(2 * M - 1)
        # Extract the sum in each diagonal
        for (idx, val) in enumerate(range(M - 1, -M, -1)):
            diag = np.diag(P, val)
            Q[idx] = np.sum(diag)

        # Compute the roots
        roots = np.roots(Q)

        # Keep the roots with radii <1 and with non zero imaginary part
        roots = np.extract(np.abs(roots) < 1, roots)
        roots = np.extract(np.imag(roots) != 0, roots)

        # Find the L roots closest to the unit circle
        distance_from_circle = np.abs(np.abs(roots) - 1)
        index_sort = np.argsort(distance_from_circle)
        component_roots = roots[index_sort[:L]]

        # extract frequencies ((note that there a minus sign since Yn are
        # defined as [y(n), y(n-1),y(n-2),..].T))
        angle = -np.angle(component_roots)

        # frequency normalisation
        angular_power = Fe * angle / (2. * np.pi)
        print(angular_power)


    # Normalize angular power
    # angular_power = angular_power/np.max(angular_power)

    ##########################################################################
    # Plot the angular power
    ##########################################################################
    # angular_power_pt = np.vstack([power * np.array([np.cos(theta), np.sin(theta)])
    #                               for power, theta in zip(angular_power, np.linspace(0, np.pi, 180))])
    # plt.plot(angular_power_pt[:, 0], angular_power_pt[
    #          :, 1], _colors[run_iterator],  alpha=0.7)
    # theta = np.linspace(0, np.pi, 180)[np.argmax(angular_power)]
    theta=angular_power
    pmax = np.max(angular_power)
    plt.plot
    plt.plot([0, pmax * np.cos(theta)], [0, pmax * np.sin(theta)],
             _colors[run_iterator], label=str(run), alpha=0.7)

#     angle = int(re.split('-|\.', run)[1])
#     runid = int(re.split('-|\.', run)[1])

#     labels.append([angle, np.linspace(0,np.pi,180)[np.argmax(angular_power)]/np.pi*180 ])

# _labels = np.vstack(labels)
# error = _labels[:,0]- _labels[:,1]
# error_traditional = np.abs(error)
# print error_traditional.mean()

##########################################################################
# Plot auxiliary stuff
##########################################################################

_, pmax1 = plt.ylim()
pmax2, pmax3 = plt.xlim()
pmax = max(pmax1, -pmax2, pmax3)
plt.ylim([0, pmax])
plt.xlim([-pmax, pmax])

for power in pmax * np.linspace(0.25, 1, 4):
    angular_power_ref = np.vstack(
        [power * np.array([np.cos(theta), np.sin(theta)]) for theta in np.linspace(0, np.pi, 180)])
    plt.plot(angular_power_ref[:, 0],
             angular_power_ref[:, 1], 'k-.', alpha=0.5)

for theta in np.linspace(0, np.pi, 19):
    plt.plot([0, pmax * np.cos(theta)],
             [0, pmax * np.sin(theta)], 'k-.', alpha=1)
plt.legend()


for theta in (ANG / 360.) * (2. * np.pi):
    plt.plot([pmax * np.cos(theta)], [pmax * np.sin(theta)], 'ko', alpha=1)
plt.legend()


if len(RUNS) == 1:
    plt.figure(figsize=(8, 4))
    plt.title('signal magnitude (' + str(run) + ')')
    for i, x in enumerate(trimmed_samples):
        plt.plot(np.abs(x), label='RX' + str(i + 1))
    plt.legend()

    t = input('')
