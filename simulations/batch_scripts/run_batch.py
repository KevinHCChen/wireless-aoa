import glob
import os
import argparse

testing = False


# generate and scp bash scripts to the eigens
def scp_eigen_bash(file_name, file_idxs, remote_host, remote_path):
    # create local file
    f = open(file_name, 'w')
    f.write('#!/usr/bin/zsh\n')
    f.write('source /home/mcrouse/.zshrc\n')
    f.write('git fetch\n')
    f.write('git checkout samepointnoisey\n')
    f.write('git pull origin samepointnoisey\n')
    f.write('python driver_base.py -d %s -s %d -e %d' % (configfile_dir.split('../')[1], file_idxs[0], file_idxs[1]))
    f.close()

    # create scp command
    scp_command = "scp %s %s:%s" % (file_name, remote_host, remote_path)

    print "SCPing the following: %s" % (scp_command)

    # scp the bash file to remote_host
    if not testing:
        os.system(scp_command)

    chmod_command = 'ssh %s "/bin/chmod u+x %s"' % (remote_host, remote_path + file_name.split('/')[-1])

    print "CHMODing the following: %s" % (chmod_command)

    if not testing:
        os.system(chmod_command)

    return remote_path + file_name.split('/')[-1]


# execute the bash scripts on the eigens
def execute_remote_bash(remote_host, bash_script):
    # create ssh command to run screen
    ssh_command = 'ssh %s "screen -d -m %s"' % (remote_host, bash_script)

    print "Executing %s" % (ssh_command)
    
    # execute ssh command
    if not testing:
        os.system(ssh_command)

    return 



if __name__ == "__main__":

    # get directory of ini files to batch out
    parser = argparse.ArgumentParser(description='Script for batching 5G Experiments to Eigens')
    parser.add_argument('--configfile_dir', '-d', dest='configfile_dir', type=str,
                        help='Which directory of config files to use')
    args = parser.parse_args()

    configfile_dir = args.configfile_dir
    # add trailing slash if needed
    if configfile_dir[-1] != '/':
        configfile_dir += '/'


    # folder to hold bash scripts (so we can see which machine is running what)
    bash_script_dir = 'eigen_bash_scripts'

    # create if doesn't exist
    if not os.path.exists(bash_script_dir):
        os.makedirs(bash_script_dir)

    # create list of all the eigens and their names
    # we will use this for bash file generation and for scp/ssh-ing
    machines = []
    for i in range(9,16):
        machines.append(('eigen%d' % (i),'mcrouse@eigen%d.int.seas.harvard.edu' % (i)))

    # get ini files from 
    files = glob.glob(configfile_dir + '*')
    files.sort()
    
    # for testing -- can delete once working
    # files = range(24)

    print "\n\nBatching the following %d files to the Eigens: \n%s\n\n" % (len(files), files)
    
    # equally divide files to the eigens
    per_machine, remainder = len(files)/len(machines), len(files)%len(machines)
    # print per_machine
    # print remainder

    print "Each Eigen will receive %d files with a remainder of %d files equally split\n" % (per_machine, remainder)


    file_idxs = []
    current_file_idx = 0
    for i, machine in enumerate(machines):
        start_file = current_file_idx
        end_file = start_file + (per_machine - 1)
        if remainder > 0:
            end_file += 1
            remainder -= 1
        file_idxs.append((start_file, end_file))
        current_file_idx = end_file + 1
    # print file_idxs


    # generate and scp custom per-eigen bash scripts to the eigens
    files_on_eigen = []
    for i in range(len(machines)):
        newfn = scp_eigen_bash(bash_script_dir + '/%s_bash_script.sh' % machines[i][0], file_idxs[i], machines[i][1], '/home/mcrouse/wireless-aoa/simulations/')
        files_on_eigen.append(newfn)

    # execute remote bash scripts on the eigens
    for i in range(len(machines)):
        execute_remote_bash(machines[i][1], files_on_eigen[i])







