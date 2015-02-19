__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import numpy as np
from matplotlib import pyplot as plt

def normalize(data):
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]

    mean = [sum(xs)/len(xs), sum(ys)/len(ys)]

    covariance_ml = [[0,0],[0,0]]
    for x in [d for d in data]:
        covariance_ml = np.add(covariance_ml, np.outer(np.subtract(x, mean), np.transpose(np.subtract(x, mean))))
    covariance_ml = np.multiply(covariance_ml, 1.0/len(data))

    L = np.linalg.cholesky(covariance_ml)

    return lambda dataset: [np.dot(np.linalg.inv(L), np.subtract(point,mean)) for point in dataset]

##################
# Plots a set of #
# data points    #
##################
def plot_points(points, ax, point_color=[1,0,0], label=''):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs,ys, color=point_color, label='Iris ' + label)
    plt.gca().legend(loc='upper right')

##################
# Plots the set  #
# of iris data   #
# points         #
##################
def plot_irises(data, labels, ax, point_colors=[[1,0,0],[0,1,0],[0,0,1]]):
    xmin = min([d[0] for d in data])
    xmax = max([d[0] for d in data])
    ymin = min([d[1] for d in data])
    ymax = max([d[1] for d in data])
    x_offset = (xmax - xmin)/10.0
    y_offset = (ymax - ymin)/10.0
    ax.set_xlim([xmin-x_offset, xmax+x_offset])
    ax.set_ylim([ymin-y_offset, ymax+y_offset])
    ax.set_title('Irises')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    label = ['setosa', 'virginica', 'versicolor']
    for i in xrange(len(point_colors)):
        plot_points([data[j] for j in xrange(len(data)) if labels[j] == i], ax, point_color=point_colors[i], label=label[i])