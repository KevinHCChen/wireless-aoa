import os, sys
import glob
import argparse

def fetch_results(srv, exp_name, out_dir):
   scp_str = "scp -r %s:wireless-aoa/simulations/experiment_results/%s\* %s/" \
             % (srv, exp_name, out_dir)
   os.system(scp_str)
   scp_str = "scp -r %s:wireless-aoa/simulations/aggregated_results/%s\* %s/" \
             % (srv, exp_name, out_dir)
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

    srv_list = ['eigen11', 'eigen12', 'eigen13', 'eigen14']

    if not os.path.exists(out_dir):
       os.makedirs(out_dir)

    for srv in srv_list:
       fetch_results(srv, exp_name, out_dir)

