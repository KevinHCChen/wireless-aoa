import numpy as np
import glob


def toa2Complex(toa, db, freq = 2.4e9):
    radian = toa*freq*2*np.pi
    return (10**(db/10.))*(np.cos(radian)+1j*np.sin(radian) )

def parseTOALog(fname):

    with open(fname) as f:
        content = f.readlines()

    # Get rid of the first two lines (header)
    content.pop(0)
    rx_num = int(content.pop(0))

    # Parse file and convert things into complex number
    signal = np.zeros(rx_num, dtype = np.complex)
    for rx in range(rx_num):
        paths = int(content.pop(0).split()[1])

        for path in range(paths):
            pathid, toa, db = [float(x) for x in content.pop(0).split()]
            signal[rx] += toa2Complex(toa, db)


    # signal is the output (complex number)
    return signal

def parseMDOALog(fname):
    try:
        with open(fname) as f:
            content = f.readlines()
    except:
        return False

    # Remove first 2 lines
    content.pop(0)
    content.pop(0)
    # the mean angles is under the Phi column, which is 6th
    angles = [float(rx_info.split()[5]) for rx_info in content]
    return np.mean(angles)

# /Users/mcrouse/Google Drive/MMWave/1800tx/AOA-Outdoor.toa.t018_06.r003.p2m
def getTxId(path):
    fname = path.split('/')[-1]
    tx_info = fname.split('.')[2]
    return int(tx_info.split('_')[0][1:])


def loadData(base_dir, file_format):

    base_dir = '/Users/mcrouse/Google Drive/MMWave/1800tx/'

    file_format = 'AOA-Outdoor.%s.t%s_06.r003.p2m'
    toa_file_list = glob.glob(base_dir + file_format % ('toa', '*'))
    mdoa_file_list = glob.glob(base_dir + file_format % ('mdoa', '*'))
    file_format = 'AOA-Outdoor.%s.t%03d_06.r003.p2m'

    phi_off = []
    label = []
    for idx, fname in enumerate(toa_file_list):
        txId = getTxId(fname)
        phi_off.append(parseTOALog(fname))
        angle =  parseMDOALog(base_dir + file_format % ('mdoa', txId))
        if angle:
            label.append(angle)
        else:
            phi_off.pop()

    return np.array(phi_off), np.array(label)



