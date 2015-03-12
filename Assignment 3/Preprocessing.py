__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import numpy as np


def normalize(data):
    #We want to sum over the columns, not the rows. Therefore, we transpose the matrix:
    mean = [sum(values)/len(values) for values in np.transpose(data)]

    diff = [np.subtract(x, mean) for x in data]
    diff_squared = [np.outer(d,np.transpose(d)) for d in diff]
    covariance_ml = reduce(lambda stack, d: np.add(stack, d), diff_squared)


    covariance_ml = np.multiply(covariance_ml, 1.0/len(data))

    L = np.linalg.cholesky(covariance_ml)

    return lambda dataset: [np.dot(np.linalg.inv(L), np.subtract(point,mean)) for point in dataset]



