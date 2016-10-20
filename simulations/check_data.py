import plotly as py
import plotly.graph_objs as go
from plotly import tools
import numpy as np
from data_generation import *
from noise_models import nonlinear_effect_function


def rescale(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def plot_data(data):
    trace = go.Heatmap(
        x=np.arange(5),
        y=np.arange(6),
        z=data.T,
        colorscale='Jet'
    )

    trace = go.Scatter(
        #x=np.tile(np.arange(6), 5),
        #y=np.repeat(np.arange(5),6),
        x=np.tile(np.linspace(-2.5,2.5, 6), 5),
        y=np.repeat(np.linspace(-1.6,1.6, 5), 6),
        mode='markers',
        marker=dict(
            color=data.T.ravel(),
            colorscale='Jet',
            size=20,
            colorbar=go.ColorBar(
                title='Colorbar',
            ),
        )

    )

    return trace

days = []
days.append( np.load('../data/numpy_data/2016-9-20_day1.npz')['all_samples'])
days.append( np.load('../data/numpy_data/2016-9-21_day2.npz')['all_samples'])
# days[1][2,:,2,0] - update this to this:
days[1][2,2,2,0] = days[1][2,0,2,0]
bs = 3

day = 1
data = days[day-1]

base_stations = [((-3.58,0), [90.]),((0, 2.726), [180.]), ((3.58,0), [270.])]
virt_x=np.tile(np.linspace(-2.5,2.5, 6), 5)
virt_y=np.repeat(np.linspace(-1.6,1.6, 5), 6)

mobiles = np.vstack((virt_x, virt_y)).T
angles = get_mobile_angles(base_stations, mobiles, 2)
angles %= 360

trace_data = []
run = 0
for bs in range(3):
    trace = plot_data(np.abs(angles[:,bs] - (180 - days[day-1][bs,run,:,:].T.ravel())))
    print angles[:,bs]
    print  days[day-1][bs,run,:,:].T.ravel()
    print angles[:,bs] - (180. - days[0][bs,run,:,:].T.ravel())
    #py.offline.plot([trace], filename="angles_err_%d.html" % (bs))

    x_sort_idx  = np.argsort(np.abs(90 - angles[:,bs]))
    exp_tmp = 180. - days[0][bs,run,:,:].T.ravel()

    trace_data.append( go.Scatter(
        x=np.abs(90. - angles[x_sort_idx,bs]),
        y=np.abs(angles[x_sort_idx, bs] - exp_tmp[x_sort_idx]),
        mode='markers',
        marker=dict(
            size=8,
            opacity=.7,
            colorscale='Jet',
            color=np.abs(angles[x_sort_idx, bs] - exp_tmp[x_sort_idx])

        )
    )
    )

k = 2
j = 1 
x = np.linspace(1/4.,1/2.,100)
multiplier_data = np.exp(np.abs(k * (x - (90/360.)))) - j
trace_data.append(
    go.Scatter(
        x=(x*360)-90,
        y=multiplier_data*.08*360
    )
)

py.offline.plot(trace_data, filename="angles_err_%d.html" % (bs))
assert False

xdim = data.shape[2]
ydim = data.shape[3]

data = rescale(data)
fig = tools.make_subplots(rows=3, cols=1)
for i in range(data.shape[0]):
    #for j in range(data.shape[1]):
    for j in range(1):
       #fig.append_trace(plot_data(data[i,j,:,:], ), i+1,j+1)
       fig.append_trace(plot_data(np.mean(data[i,:,:,:], axis=0)), i+1,j+1)


py.offline.plot(fig, filename='checkangle_day_runAvg_%d' %(day))

