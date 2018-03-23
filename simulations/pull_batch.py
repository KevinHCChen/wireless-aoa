import os, sys
import glob
import argparse

def fetch_results(srv, exp_name, out_dir):
   scp_str = "scp -r %s:wireless-aoa/simulations/experiment_results/%s\* %s/" \
             % (srv, exp_name, out_dir)
   #os.system(scp_str)
   scp_str = "rsync -ravz %s:wireless-aoa/simulations/aggregated_results/%s\* %s/" \
             % (srv, exp_name, out_dir)
   print scp_str
   os.system(scp_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver for 5G Experiments')
    parser.add_argument('--out_dir', '-o', dest='out_dir', type=str,
                       help='Directory to store pulled files')
    parser.add_argument('--exp_name', '-e', dest='exp_name', type=str,
                       help='Directory to store pulled files')

    args = parser.parse_args()
    out_dir = args.out_dir
    exp_name = args.exp_name

    assert out_dir, "Must specify output directory!"
    print out_dir

    #srv_list = ['eigen11', 'eigen12', 'eigen13', 'eigen14']
    srv_list = ['eigen%d' % (i) for i in range(9,16)]

    exp_names = ['nonsensenoise', 'spuriousnoise', 'nooutputnoise']
    exp_names = ['angledependentdistnoise_02', 'angledependentdistnoise_03']
    exp_names = ['gaussiannoise_0p03']
    exp_names = ['samepointnoisey_100innerloop_uniform']
    exp_names = ['nonsensenoise_40iters', 'nooutputnoise_40iters']
    exp_names = ['grid_training_initial_exploration_100iters']
    exp_names = ['4bs_nonoise', '4bs_nooutput_cv0', '4bs_nonsense_0r1',
                 '4bs_spurious_0r1_cr0p1', '4bs_angledependent_0p01',
                  '4bs_gaussian_0p01']
    #exp_names = ['Exp1']
    exp_names = ['experiment*']
    exp_names = ['3bs_gaussian']
    exp_names = ['exp3']
    exp_names = ['exp1_perfect']
    exp_names = ['exp2_gaussian_median']
    exp_names = ['exp3_angledependent_median']


    for srv in srv_list:
      for exp_name in exp_names:
        #out_dir = exp_name
        if not os.path.exists(out_dir):
          os.makedirs(out_dir)
        fetch_results(srv, exp_name, out_dir)

