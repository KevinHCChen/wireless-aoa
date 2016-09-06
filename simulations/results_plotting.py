import numpy as np
import glob
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import argparse

def plot_increased_training(data, out_file, exp_name=None):
    nn_sizes = np.sort(data['NN__network_size'].unique())
    net_types = np.sort(data['NN__type'].unique())

    if exp_name:
        data = data[data['exp_details__name'].str.contains(exp_name)]
        print data.shape
    data_list = []
    for n_type in net_types:
        select = data[data['NN__type'] == n_type]
        nn_sizes = np.sort(select['NN__network_size'].unique())
        for nn_size in nn_sizes:
            crit_str = nn_size.replace('[', '\[').replace(']', '\]')
            select2 = select[select['NN__network_size'].str.match(crit_str)]
            select2 = select2.sort_values(by=['data__num_pts'])
            if select2.shape[0] > 0:
                y = select2['mean_err']
                x = select2['data__num_pts']
                data_list.append(go.Scatter(
                    x = x,
                    y = y,
                    #error_y=dict(
                    #            type='data',
                    #            array=select2['std_err']**2,
                    #            visible=True
                    #        ),
                    name='%s-%s' %(n_type, nn_size)
                ))


    layout = dict(title = "%s - Increasing Training Size under Base and Structured Models" % (exp_name),
                  xaxis = dict(title="Training Data Size"),
                  yaxis = dict(title="MSE")
                )
    fig = dict(data=data_list, layout=layout)

    py.offline.plot(fig, filename=out_file)

def load_results(res_dir):
    fns = glob.glob(res_dir+"/*.csv")
    print fns
    data = [pd.read_csv(fn) for fn in fns]
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plotting for 5G Experiments')
    parser.add_argument('--results_dir', '-d', dest='results_dir', type=str,
                        help='Which config file to use')

    args = parser.parse_args()

    exp_names = ['nonsensenoise', 'spuriousnoise', 'nooutputnoise']
    #exp_names = ['gaussiandistnoise']
    #exp_names = ['angledependentdistnoise']
    #exp_names = ['no_noise']

    data_l = []

    for exp in exp_names:
        data_l += load_results(exp)
    #data_l = load_results(args.results_dir)
    print len(data_l)
    data = pd.concat(data_l, ignore_index=True)
    print data.shape

    for exp in exp_names:
        base_dir = "summary_plots/"
        exp_name = "%s" % (exp)
        plot_increased_training(data, base_dir + exp_name + "_increasing_training.html", 'wnoise')

