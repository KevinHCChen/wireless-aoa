
import numpy as np


fname = '/Users/kevin/Downloads/toa2.p2p'


def toa2Complex(toa, db, freq = 2.4e9):
    radian = toa*freq*2*np.pi
    return (10**(db/10.))*(np.cos(radian)+1j*np.sin(radian) )

with open(fname) as f:
    content = f.readlines()

# Get rid of the first two lines (header)
content.pop(0)
rx_num = int(content.pop(0))

# Parse file and convert things into complex number
signal = np.zeros(rx_num, dtype = np.complex)
for rx in range(rx_num):
    paths = int(content.pop(0).split(' ')[1])

    for path in range(paths):
        pathid, toa, db = [float(x) for x in content.pop(0).split(' ')]
        signal[rx] += toa2Complex(toa, db)


# signal is the output (complex number)
