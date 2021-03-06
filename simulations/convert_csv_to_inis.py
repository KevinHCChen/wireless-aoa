import csv
import random
import os
import argparse

parser = argparse.ArgumentParser(description='Driver for Config Generation')
parser.add_argument('--config_csv', '-d', dest='config_name', type=str,
                    help='Which csv file to use to generate inis')
args = parser.parse_args()

config_name = args.config_name
dir_name = 'batch_exp_configs/%s/' % (config_name)
if not os.path.exists(dir_name):
        os.makedirs(dir_name)

with open('experiment_set_csvs/%s.csv' % (config_name), 'rU') as f:
    reader = csv.reader(f)
    first = True
    num_files_gen = 0
    for row in reader:
        if first:
            questions_row = row
            questions = row[5:-7]
            first = False
        else:
            f = open(dir_name + '%s.ini' % (row[0]), 'w')

            f.write('[exp_details]\n')
            f.write('setname: %s\n' % (row[15]))
            f.write('name: %s\n' % (row[0]))
            f.write('description: %s\n' % ('NA'))
            f.write('save: %s\n' % ('True'))
            f.write('interactive: %s\n' % ('True'))
            f.write('num_iterations_per_setting: %s\n' % (row[14]))

            f.write('\n')
            f.write('[NN]\n')
            f.write('type: %s\n' % (row[3]))
            f.write('network_size: %s\n' % (row[4]))
            f.write('n_epochs: %d\n' % (500))
            f.write('batchsize: %d\n' % (200))
            f.write('take_max: %s\n' % ('True'))
            f.write('# Epsilon is in terms of squared L2\n')
            if row[7] != '':
                f.write('epsilon: %s\n' % (row[7]))
            else:
                f.write('epsilon: .02\n')

            f.write('\n')
            f.write('[data]\n')
            if len(row) >= 17:
                f.write('data_dist: %s\n' % (row[16]))
            f.write('num_pts: %s\n' % (row[1]))
            f.write('ndims: %s\n' % (row[6]))
            f.write('num_stations: %s\n' % (row[2]))
            f.write('sphere_r: %s\n' % (4))
            f.write('bs_type: %s\n' % (row[8]))

            f.write('\n')
            f.write('[noise]\n')
            f.write('addnoise_train: %s\n' % (row[9]))
            f.write('addnoise_test: %s\n' % (row[10]))
            f.write('noise_model: %s\n' % (row[11]))
            f.write('noise_params: %s\n' % (row[12]))
            f.write('bases_to_noise: %s\n' % (row[13]))

            f.close()

        num_files_gen += 1


print 'Automatic INI File Generation Completed!'
print '%d files automatically generated' % (num_files_gen)
print 'This saved you %d minutes' % (num_files_gen*17)
print 'Congrats!!'




