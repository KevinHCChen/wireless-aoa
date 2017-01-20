import numpy as np
import glob

num_rx = 4
grid_bs = 3
grid_x = 6
grid_y = 5
grid_r = 3
grid_rx = 4

#save_fn_name = '2016-9-20_day1_rawPhase_norm.npz'
#save_fn_name = '2016-9-21_day2_rawPhase_norm.npz'
#save_fn_name = '2016-9-21_day2_rawPhase_median.npz'
#save_fn_name = '2016-9-24_day3_inside_rawPhase.npz'
#save_fn_name = '2016-9-25_day4_inside_rawPhase.npz'
#save_fn_name = '2017-1-19_indoorLounge_morning.npz'
save_fn_name = '2017-1-19_indoorLounge_afternoon.npz'

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


#files = glob.glob('../data/2016-9-20_day1_outside/*rx1*')
#files = glob.glob('../data/2016-8-21_day2_outside/*rx1*')
#files = glob.glob('../data/2016-8-21_day2_outside/*rx1*')
#files = glob.glob('../data/2016-9-24_day3_inside/*rx1*')
#files = glob.glob('../data/2016-9-25_day4_inside/*rx1*')
#files = glob.glob('../data/2017-1-19_indoorLounge_morning/*rx1*')
files = glob.glob('../data/2017-1-19_indoorLounge_afternoon/*rx1*')
all_samples = np.zeros((grid_bs, grid_r, grid_x, grid_y, grid_rx ))

for f in files:
    print f
    current_x = int(f[-11])
    current_y = int(f[-8])
    current_r = int(f[-5])
    current_bs = int(f[-14])


    f_sp = f.split('rx1')
    trimmed_samples = [ readSamples(f_sp[0]+'rx'+str(rx+1)+f_sp[1]) for rx in range(num_rx) ] 

    synced_samples, synced_calibration_samples =  calibration(trimmed_samples)


    phase_offset = [np.mean(np.absolute(synced_samples[i])) for i in range(num_rx)]
    #phase_offset = [np.mean(np.linalg.norm(synced_samples[i])) for i in range(num_rx)]
    #phase_offset = [np.median(np.absolute(synced_samples[i])) for i in range(num_rx)]
    #phase_offset = [np.mean(np.angle(synced_samples[i])) for i in range(num_rx)]
    aggregated_samples = np.vstack(phase_offset)

    all_samples[current_bs-1, current_r-1, current_x, current_y, :] = phase_offset
    #all_samples[current_bs-1, current_r-1, current_x, current_y, :] = aggregated_samples

 
sf = open(save_fn_name, 'w')
np.savez(sf, all_samples=all_samples)
sf.close()
