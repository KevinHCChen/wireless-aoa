#!/bin/bash

#DELL_TX=10.243.42.181
#DELL_RX=10.243.42.180
DELL_TX=192.168.12.63
DELL_RX=192.168.12.61

# REF=/root/aoa/random4.dat
REF=/root/aoa/ones1000.dat
FREQ=916000000
RATE=500000
RX=4
BURST=2

# sudo ifconfig en5 192.168.12.77 up


#   
#   scp 4rx_to_file.cpp root@192.168.12.61:/root/aoa/
#   cp 4rx_to_file.cpp ~/UHD-install/EttusResearch-UHD-Mirror-e625e89/host/examples/
#   cd ~/UHD-install/EttusResearch-UHD-Mirror-e625e89/host/examples/; make
#   cat ~/UHD-install/EttusResearch-UHD-Mirror-e625e89/host/examples/CMakeLists.txt 
#   cp /root/UHD-install/EttusResearch-UHD-Mirror-e625e89/host/examples/4rx_to_file ~/aoa/

echo "Starting RX - Run $1"
ssh root@$DELL_RX "sh -c 'nohup  /root/aoa/4rx_to_file \\
--freq=$FREQ --rate=$RATE --gain=25 --run=$1 --runtime 4 \\
--ip1=addr=192.168.11.14 --ip2=addr=192.168.11.15  --ip3=addr=192.168.11.16 --ip4=addr=192.168.11.18 \\
> /dev/null 2>&1 &'"

sleep 2.5
# sleep 4.5
echo "Start TX"
ssh root@$DELL_TX "/root/aoa/start_tx.sh $FREQ $RATE $REF $REF" 
sleep 3
echo "Done measuring!"

echo "Pruning samples"
ssh root@$DELL_RX "python /root/aoa/trim_data.py $1 $BURST $RX"


echo "Pulling data"
scp root@$DELL_RX:/root/aoa/data/t*_$1.dat ../data/testdata/
echo "All done!"

python AOA.py -- --run=$1 --rx=$RX
echo "python AOA.py -- --run=$1 --rx=$RX"
