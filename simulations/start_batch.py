import os, sys
import glob


srv_dir = "/home/mcrouse/wireless-aoa/simulations"

def start_run(srv, script_name):
    run_str = 'ssh %s "screen -d -m %s/%s"' %(srv_dir, srv, script_name)
    print run_str
    os.system(run_str)

def gen_batch_file(fn, server_batch_dir, file_idxs):
    fn = ""
    with fh as open(fn, 'w'):
       fh.write('#!/usr/bin/zsh\n')
       fh.write('source /home/mcrouse/.zshrc\n')
       fh.write('python driver_base.py -b %s -s %d -e %d' % (server_batch_dir, file_idxs[0], file_idxs[1]))

    return fn

def copy_batch_file(fn, srv, srv_path):
    cmd_str = "scp %s %s:%s/" % (fn, srv, srv_path)



if __name__ == "__main__":
    pairs = {'eigen11' : [0,20]}
    for srv, idx in pairs.iteritems():
        gen_batch_file(fn, batch_dir, idx)
        copy_batch_file(fn, srv, srv_path)
        start_run('eigen11', 'test_batch.sh')
