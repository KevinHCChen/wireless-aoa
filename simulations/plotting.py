import plotly as py
import plotly.graph_objs as go
from plotly import tools
import numpy as np

color_vector = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
color = ['red', 'green', 'blue', 'c', 'm', 'y', 'k']

def plotStations(baseStations):
    traces = []
    for i, bs in enumerate(baseStations):
        #plt.plot(bs[0][0], bs[0][1], marker=(4, 0, bs[1]), markersize=40)
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1][0]))
        eq = to_coef(slope, cords)
        traces.append( plotLine(eq, bs[0], bs[1][0], i) )
    return traces


def plotStations3D(baseStations, fig):
    for i, bs in enumerate(baseStations):
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1][0]))
        eq = to_coef(slope, cords)
        plotPlane(eq, bs[0], bs[1][0], fig, color_vector[i%len(color_vector)])


def plotLine(eq, center, ang, idx):
    if ang % 180 < 45 or ang % 180 > 135:
        x1 = center[0] + .2
        x2 = center[0] - .2
        y1 =  (eq[1] - eq[0][0]*x1)/(eq[0][1])
        y2 =  (eq[1] - eq[0][0]*x2)/(eq[0][1])
    else:
        y1 = center[1] + .2
        y2 = center[1] - .2
        x1 =  (eq[1] - eq[0][1]*y1)/(eq[0][0])
        x2 =  (eq[1] - eq[0][1]*y2)/(eq[0][0])


    trace = go.Scatter(
        x=[x1,x2],
        y=[y1,y2],
        line = dict(
            color = color[idx%len(color)],
            width = 10
        )
        #mode='markers',
        #marker=dict(
        #    size=20,
        #)
    )

    return trace


def plotPlane(eq, center, ang, fig, color): 
    if ang % 180 < 45 or ang % 180 > 135:
        x1 = center[0] + .2
        x2 = center[0] - .2
        y1 =  (eq[1] - eq[0][0]*x1)/(eq[0][1])
        y2 =  (eq[1] - eq[0][0]*x2)/(eq[0][1])
    else:
        y1 = center[1] + .2
        y2 = center[1] - .2
        x1 =  (eq[1] - eq[0][1]*y1)/(eq[0][0])
        x2 =  (eq[1] - eq[0][1]*y2)/(eq[0][0])
    x = [x1,x2,x2,x1]
    y = [y1,y2,y2,y1]
    z = [-1,-1,1,1]
    verts = [zip(x,y,z)]
    ax = fig.gca(projection='3d')
    p3c = Poly3DCollection(verts)
    p3c.set_facecolor(color)
    ax.add_collection3d(p3c)


def to_coef(m, p):
    #conversion from point/slope form of line to standard form. Do on paper if confused.
    const = p[1]-m*p[0]
    #return the left side coefficients and the right side constant. these represent the equation.
    return ([-m, 1], const)

def rescale(data):
    #data = np.nan_to_num(data) # TODO: FIX THIS!!!! YIKES
    return (data - np.min(data))/(np.max(data) - np.min(data))

def plot_scatter(positions, error, title, to_rescale=False):

    if to_rescale:
        trace = go.Scatter(
            x=positions[:,0],
            y=positions[:,1],
            mode='markers',
            marker=dict(
                size=8,
                color=rescale(error),
                colorscale='Jet',
                colorbar=go.ColorBar(
                    title='Colorbar',
                    #yanchor='bottom'
                    ),
                showscale=True,
                opacity=0.8
            )
        )
    else:
        error_mod = error
        error_mod[np.where(error > 1.)[0]] = 1.
        trace = go.Scatter(
            x=positions[:,0],
            y=positions[:,1],
            mode='markers',
            marker=dict(
                size=8,
                color=error,
                colorscale='Jet',
                cmin=0.,
                cmax=1.,
                showscale=True,
                colorbar=go.ColorBar(
                    title='Colorbar',
                    #yanchor='top'
                    ),
                opacity=0.8
            )
        )

    return trace

def plot_scatter3D(positions, error, title):

    trace = go.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode='markers',
        marker=dict(
            size=8,
            color=rescale(error),
            #color=error,
            colorscale='Jet',
            showscale=True,
            opacity=0.5
        )
    )


    return trace

# positions (x,y,z?) as numpy array
def plot_error(true_pos, predicted_pos, error, bases, title, saveflag, dir_name, iter_number):
    # 2D case
    if true_pos.shape[1] == 2:
        fig = tools.make_subplots(rows=1, cols=2)
        trace = plot_scatter(true_pos, error, title)
        fig.append_trace(trace, 1,1)
        #trace = plot_scatter(true_pos, error, title, True)
        #fig.append_trace(trace, 2,1)

        trace = plot_scatter(predicted_pos, error, title)
        fig.append_trace(trace, 1,2)
        #trace = plot_scatter(predicted_pos, error, title, True)
        #fig.append_trace(trace, 2,2)
        ax_range = [-4.5,4.5]
        fig['layout']['xaxis1'].update( range=ax_range)
        fig['layout']['xaxis2'].update( range=ax_range)
        #fig['layout']['xaxis3'].update( range=ax_range)
        #fig['layout']['xaxis4'].update( range=ax_range)
        fig['layout']['yaxis1'].update( range=ax_range)
        fig['layout']['yaxis2'].update( range=ax_range)
        #fig['layout']['yaxis3'].update( range=ax_range)
        #fig['layout']['yaxis4'].update( range=ax_range)
        fig['layout'].update(showlegend=False)

        for t in plotStations(bases):
            fig.append_trace(t, 1,1)
            #fig.append_trace(t, 1,2)

        if saveflag:
            py.offline.plot(fig, filename=dir_name + 'error-fig-iteration%d.html' % (iter_number))

    # 3D case
    if true_pos.shape[1] == 3:
        fig = tools.make_subplots(rows=1, cols=2, specs=[[{'is_3d': True}, {'is_3d': True}]])
        data = plot_scatter3D(true_pos, error, title)
        fig.append_trace(data, 1,1)

        data = plot_scatter3D(predicted_pos, error, title)
        fig.append_trace(data, 1,2)
        fig['layout']['scene1']['aspectmode'] = 'cube'
        fig['layout']['scene2']['aspectmode'] = 'cube'


        if saveflag:
            py.offline.plot(fig, filename=dir_name + 'error-fig-iteration%d.html' % (iter_number))

