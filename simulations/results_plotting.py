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
        data = data[data['exp_details__setname'].str.contains(exp_name)]
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
                    #y = np.sqrt(y),
                    name='%s-%s' %(n_type, nn_size)
                ))


    layout = dict(title = "%s - Increasing Training Size under Base and Structured Models" % (exp_name),
                  xaxis = dict(title="Training Data Size"),
                  yaxis = dict(title="MSE")
                )
    fig = dict(data=data_list, layout=layout)
    py.offline.plot(fig, filename=out_file)

def plot_increasing_stations(data, out_file, exp_name=None):
    nn_sizes = np.sort(data['NN__network_size'].unique())
    net_types = np.sort(data['NN__type'].unique())
    train_size = np.sort(data['data__num_pts'].unique())

    if exp_name:
        data = data[data['exp_details__setname'].str.contains(exp_name)]
        data = data[data['NN__type'] == net_types[0]]
        print data.shape
    data_list = []

    for n_pts in train_size:
        print n_pts
        select = data[data['data__num_pts'] == n_pts]
        nn_sizes = np.sort(select['NN__network_size'].unique())
        for nn_size in nn_sizes:
            crit_str = nn_size.replace('[', '\[').replace(']', '\]')
            select2 = select[select['NN__network_size'].str.match(crit_str)]
            select2 = select2.sort_values(by=['data__num_stations'])
            if select2.shape[0] > 0:
                y = select2['mean_err']
                x = select2['data__num_stations']
                data_list.append(go.Scatter(
                    x = x,
                    y = y,
                    #y = np.sqrt(y),
                    name='Model: %s-%s - Training Size: %s' % (net_types[0],nn_size, n_pts)
                ))


    layout = dict(title = "%s - Increasing Number of Base Stations " % (exp_name),
                  xaxis = dict(title="Number of Base Stations"),
                  yaxis = dict(title="MSE")
                )
    fig = dict(data=data_list, layout=layout)
    py.offline.plot(fig, filename=out_file)


def plot_methods(data, out_file):
    methods = [1,2]

    models = data['NN__type'].unique()
    data_list = []
    for model in models:
        for method in methods:
            selected = data[(data['data__noiseyexperimentnumber'] == method) &
                            (data['NN__type'] == model)]
            selected = selected.sort_values(by=['data__numsamplesperpoints'])
            data_list.append(go.Scatter(
                #x=selected['data__numsamplesperpoints'].unique(),
                x=selected['data__numsamplesperpoints'],
                y=selected['mean_err'],
                name='%s - Method %i' % (model, method)
                )
            )

    layout = dict(title = "Noisy Point Methods",
                xaxis = dict(title="Number Averaged Points"),
                yaxis = dict(title="MSE")
                )

    fig = dict(data=data_list, layout=layout)
    py.offline.plot(fig, filename=out_file+'.html')

def load_results(res_dir):
    fns = glob.glob(res_dir+"/*.csv")
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
    exp_names = ['angledependentdistnoise_02', 'angledependentdistnoise_03']
    exp_names = ['gaussiannoise_0p03']
    #exp_names = ['results/samepointnoisey']
    #exp_names = ['samepointnoisey']
    #exp_names = ['samepointnoisey_100innerloop_uniform']
    exp_names = ['nonsensenoise_40iters', 'nooutputnoise_40iters']
    exp_names = ['uniform_grid_training_nonoise']
    exp_names = ['grid_training_initial_exploration_100iters']
    exp_names = ['4bs_nonoise', '4bs_nooutput_cv0', '4bs_nonsense_0r1',
                 '4bs_spurious_0r1_cr0p1', '4bs_angledependent_0p01',
                  '4bs_gaussian_0p01']

    exp_names = ['4bs_nooutput_cv0_Explore']
    exp_names = ['structured_500_nooutput_bs3', 'structured_500_nooutput_bs4']
    exp_names = ['increase_base_stations']

    data_l = []

    for exp in exp_names:
        data_l += load_results(exp)
    #data_l = load_results(args.results_dir)
    print len(data_l)
    data = pd.concat(data_l, ignore_index=True)
    print data.shape

    base_dir = "summary_plots/"
    # plot increased training
    if False:
        for exp in exp_names:
            exp_name = "%s" % (exp)
            plot_increased_training(data, base_dir + exp_name + "_increasing_training.html", exp_name)
    if True:
        for exp in exp_names:
            exp_name = "%s" % (exp)
            plot_increasing_stations(data, base_dir + exp_name + "_increasing_base_stations.html", exp_name)

    if False:
        plot_methods(data, base_dir + "methods_increasing_train")

