import matplotlib.pyplot as plt
import numpy as np

def plotStations(baseStations, station_len):
    for bs in baseStations:
        #plt.plot(bs[0][0], bs[0][1], marker=(4, 0, bs[1]), markersize=40)
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1][0]))
        eq = to_coef(slope, cords)
        plotLine(eq, bs[0], bs[1][0])


def plotStations3D(baseStations, station_len, fig):
    for bs in baseStations:
        cords = np.array(bs[0])
        slope = np.tan(np.radians(bs[1]))
        eq = to_coef(slope, cords)
        plotPlane(eq, bs[0], fig)


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
    if true_pos.shape[1] == 2:
        plt.figure()
        plt.subplot(2,1,1)
        plot_scatter(true_pos, error, title)
        plotStations(bases, 2)
        

        #plt.figure()#;plt.clf()
        plt.subplot(2,1,2)
        plot_scatter(predicted_pos, error, title)
        plotStations(bases, 2)
        fig = plt.gcf() 
        fig.set_size_inches(6.5, 10.5, forward=True)

        plt.savefig(dir_name + 'error_fig.png', format = 'png')

        plt.show()



    if true_pos.shape[1] == 3:
        print "Not implemented yet"

