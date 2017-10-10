import numpy as np
import struct
import matplotlib.pyplot as plt
import sys

syms = [[1,0], [-1,0], [0,1], [0,-1]]
#syms = [[1,0], [-1,0]]

num_reps = 1000

tk = ""
for i in range(num_reps):
    for sym in syms:
        tk = tk + struct.pack("<ff", sym[0], sym[1])



fh = open('transmit_streams/cross.dat', 'wb')
fh.write(tk)
fh.close()