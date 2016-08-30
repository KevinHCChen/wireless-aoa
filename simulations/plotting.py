import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

color_vector = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def plotStations(baseStations):
    for bs in baseStations:
        #plt.plot(bs[0][0], bs[0][1], marker=(4, 0, bs[1]), markersize=40)
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1][0]))
        eq = to_coef(slope, cords)
        plotLine(eq, bs[0], bs[1][0])


def plotStations3D(baseStations, fig):
    for i, bs in enumerate(baseStations):
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1][0]))
        eq = to_coef(slope, cords)
        plotPlane(eq, bs[0], bs[1][0], fig, color_vector[i%len(color_vector)])


def plotLine(eq, center, ang):
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
    plt.plot([x1,x2], [y1,y2], '-', linewidth=10., markersize=12)


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


def plot_scatter(positions, error, title):
    plt.scatter(positions[:,0], positions[:,1], c=error)
    
    plt.colorbar()
    plt.ylim([-5,5])
    plt.xlim([-5,5])
    plt.title(title)
    plt.clim([0,1])

    # plt.axis('equal')


# positions (x,y,z?) as numpy array
def plot_error(true_pos, predicted_pos, error, bases, title, saveflag, dir_name):
    # 2D case
    if true_pos.shape[1] == 2:
        plt.figure()
        plt.subplot(2,1,1)
        plot_scatter(true_pos, error, title)
        plotStations(bases)

        #plt.figure()#;plt.clf()
        plt.subplot(2,1,2)
        plot_scatter(predicted_pos, error, title)
        plotStations(bases)
        fig = plt.gcf() 
        fig.set_size_inches(6.5, 10.5, forward=True)

        if saveflag:
            plt.savefig(dir_name + 'error_fig.png', format = 'png')

        # plt.show()

    # 3D case
    if true_pos.shape[1] == 3:
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(true_pos[:,0], true_pos[:,1], true_pos[:,2], c=error, alpha=0.1)
        ax.set_xlabel('X Plane')
        ax.set_ylabel('Y Plane')
        ax.set_zlabel('Z Plane')
        plotStations3D(bases, fig)
        ax.set_ylim((-6,6))
        ax.set_xlim((-6,6))
        ax.set_zlim((-6,6))
        plt.title("Ground Truth, Num Stations: %d" % (3))
        # plt.show()
        if saveflag:
            plt.savefig(dir_name + 'error_true_fig.png', format = 'png')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(predicted_pos[:,0], predicted_pos[:,1], predicted_pos[:,2], c=error)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plotStations3D(bases, fig)
        ax.set_ylim((-6,6))
        ax.set_xlim((-6,6))
        ax.set_zlim((-6,6))
        plt.title("Predicted, Num Stations: %d" % (3))
        # plt.show()
        if saveflag:
            plt.savefig(dir_name + 'error_predicted_fig.png', format = 'png')

