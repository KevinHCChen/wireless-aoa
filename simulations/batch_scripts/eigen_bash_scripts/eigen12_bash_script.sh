#!/usr/bin/zsh
source /home/mcrouse/.zshrc
git checkout master
git pull origin master
python driver_base.py -d batch_exp_configs/exp2_gaussian_median/ -s 14 -e 17