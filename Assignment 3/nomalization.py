__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import numpy as np


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

def mean(data):
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]

    return [sum(xs)/len(xs), sum(ys)/len(ys)]

def covariance(data, mean):
    covariance_ml = [[0,0],[0,0]]
    for x in [d for d in data]:
        covariance_ml = np.add(covariance_ml, np.outer(np.subtract(x, mean), np.transpose(np.subtract(x, mean))))

    return covariance_ml