import plotly as py
import plotly.graph_objs as go
from plotly import tools
import numpy as np



day_1 = np.load('../data/numpy_data/2016-9-20_day1.npz')['all_samples']
day_2 = np.load('../data/numpy_data/2016-9-21_day2.npz')['all_samples']
bs = 3

def plot_data(data):
	trace = go.Heatmap(
		x=np.arange(5),
		y=np.arange(6),
		z=data.T,
		colorscale='Jet'
	)

	return trace

xdim = day_1.shape[2]
ydim = day_1.shape[3]

data = []

# data.append(plot_data(np.tile(np.arange(xdim),ydim), np.repeat(np.arange(ydim), xdim), day_1[0,0,:,:].ravel()))
data.append(plot_data(day_1[bs-1,0,:,:]))

py.offline.plot(data, filename='checkangle_bs%d' % (bs))




