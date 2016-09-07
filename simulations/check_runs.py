import os


for i in range(9,16):
  print "eigen %i" % (i)
  os.system("ssh eigen%d screen -ls" % (i))
