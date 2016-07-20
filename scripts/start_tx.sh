#!/bin/bash



# Input parameters are frequency. signal_ref, signal_tx
# 916000000
# RATE=1000000
# FILE_NAME=/root/aoa/ones1000.dat
# FILE_NAME=/root/aoa/random4.dat

echo "Sending from Reference"
#/root/aoa/tx_samples_from_file --args "addr0=192.168.10.12" --freq 916000000 --rate 1000000 --gain 25 --file /root/aoa/random4.dat
/root/aoa/tx_samples_from_file --args "addr0=192.168.10.12" --freq $1 --rate $2 --gain 25 --file $3

echo "Sending from Target/Tx"
/root/aoa/tx_samples_from_file --args "addr0=192.168.10.17" --freq $1 --rate $2 --gain 25 --file $4
