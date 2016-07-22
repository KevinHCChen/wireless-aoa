#!/bin/bash

#DELL_TX=10.243.42.181
#DELL_RX=10.243.42.180
DELL_TX=192.168.12.63
DELL_RX=192.168.12.61

# REF=/root/aoa/random4.dat
REF=/root/aoa/ones1000.dat
FREQ=916000000
RATE=500000
RX=7
BURST=2

# sudo ifconfig en5 192.168.12.77 up



echo "Starting RX - Run $1"
ssh root@$DELL_RX "sh -c 'nohup  /root/aoa/7rx_to_file --freq=$FREQ --rate=$RATE --gain=25 --run=$1 --runtime 4 > /dev/null 2>&1 &'"


sleep 4.5
echo "Start TX"
ssh root@$DELL_TX "/root/aoa/start_tx.sh $FREQ $RATE $REF $REF" 
sleep 3
echo "Done measuring!"

echo "Pruning samples"
ssh root@$DELL_RX "python /root/aoa/trim_data.py $1 $BURST $RX"


echo "Pulling data"
scp root@$DELL_RX:/root/aoa/data/t*_$1.dat ../data/testdata/
echo "All done!"

ipython AoA.py -- --run=$1 --rx=$RX
echo "ipython AoA.py -- --run=$1 --rx=$RX"
