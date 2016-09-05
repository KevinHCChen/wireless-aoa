import os

nodes = [11,12,13,14]
for i in nodes:
  os.system("ssh eigen%d screen -ls" % (i))
