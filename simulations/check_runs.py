import os

#nodes = [11,12,13,14]

for i in range(9,16):
  print "eigen %i" % (i)
  os.system("ssh eigen%d screen -ls" % (i))
