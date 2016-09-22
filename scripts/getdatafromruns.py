import numpy as np
import glob

# from AOA import calibration, readSamples, genSteeringVectors


AOA_method = 'Correlation'
num_rx = 4
grid_bs = 3
grid_x = 6
grid_y = 5
grid_r = 3

save_fn_name = '2016-9-20_day1.npz' 

files = glob.glob('../data/2016-9-20_day1_outside/*rx1*')


def calibration(trimmed_samples):
    # This assumes the first half of samples are coming from reference Tx

    # separate calibration signal
    calibration_samples = [ x[:x.size/2] for x in trimmed_samples ]
    samples = [ x[x.size/2:] for x in trimmed_samples ]

    calib_ang_diff = [angleDiff( calibration_samples[i+1], calibration_samples[0] ) for i in range(num_rx-1) ]
    calib_ang_diff_avg = [0]+[ np.median(x[x.size/4:x.size*3/4]) for x in calib_ang_diff]

    synced_calibration_samples = [ x*angle2c(-y) for x,y in zip(calibration_samples,calib_ang_diff_avg)]
    synced_samples = [ x*angle2c(-y) for x,y in zip(samples,calib_ang_diff_avg)]

    return synced_samples, synced_calibration_samples


def readSamples(filename, size = -1):
    dat = np.fromfile(open(filename, 'rb'), dtype=np.float32)
    # print dat
    # print dat.shape
    if size == -1:
        size = len(dat)
    dat = dat[:size*2]
    dat_complex = dat[0::2] + dat[1::2]*1j
    return dat_complex


def genSteeringVectors():
    return np.vstack( [ steeringVector(theta) for theta in np.linspace(0,np.pi,180) ] ).T

def angleDiff(signal1, signal2):
    return np.mod(np.angle(signal1)-np.angle(signal2)+np.pi, 2*np.pi)-np.pi
def angle2c(theta):
    return 1j*np.sin(theta)+np.cos(theta)

dRxRx = -0.15
dRxTx = 2.44
freq = 916e6
speedoflight = 3e8

def steeringVector(theta, d_rx=dRxRx, d_txrx=dRxTx, frequency=freq):
    rx_loc = np.vstack([[rx*d_rx,0] for rx in range(num_rx)])
    rx_loc -= rx_loc.mean(axis=0)
    tx_loc = np.array([d_txrx*np.cos(theta), d_txrx*np.sin(theta)])

    ant_dist = np.array([ np.linalg.norm(tx_loc-x ) for x in rx_loc])
    ant_dist -= ant_dist[0]
    phase_diff = ant_dist*2*np.pi*frequency/speedoflight

    return angle2c( phase_diff )

all_samples = np.zeros((grid_bs, grid_r, grid_x, grid_y, ))


for f in files:
    print f
    current_x = int(f[-11])
    current_y = int(f[-8])
    current_r = int(f[-5])
    current_bs = int(f[-14])


    f_sp = f.split('rx1')
    trimmed_samples = [ readSamples(f_sp[0]+'rx'+str(rx+1)+f_sp[1]) for rx in range(num_rx) ] 
    # trimmed_samples = [ readSamples(args.path+'trimmed_rx'+str(rx+1)+'_'+str(run)+'.dat') for rx in range(args.rx) ] 
    synced_samples, synced_calibration_samples =  calibration(trimmed_samples)
    steeringvectors = genSteeringVectors()
    aggregated_samples = np.vstack(synced_samples)

    if AOA_method == 'Correlation':

        # Correlation approach to determine power coming from each angle    
        angular_power = np.median(np.abs(np.dot( np.conjugate(steeringvectors).T, aggregated_samples)), axis=1)

    elif AOA_method == 'Music':

        # the MUSIC approach
        covariance = np.dot(aggregated_samples, np.conjugate(aggregated_samples).T)/aggregated_samples.shape[1]
        eigvalue, eigvector = np.linalg.eig(covariance)

        noisespace = eigvector[:,np.argsort(eigvalue)[:-1]]
        noisepower = np.linalg.norm( np.dot( np.conjugate(noisespace).T, steeringvectors), axis=0)
        angular_power = 1./noisepower


    # pmax = np.max(angular_power)
    # print "PMAX: ", np.rad2deg(pmax)
    theta = np.linspace(0,np.pi,180)[np.argmax(angular_power)]
    all_samples[current_bs-1, current_r-1, current_x, current_y, ] = np.rad2deg(theta)

print all_samples

sf = open(save_fn_name, 'w')
np.savez(sf, all_samples=all_samples)
sf.close()
