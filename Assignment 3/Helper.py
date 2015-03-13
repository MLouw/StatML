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

def save_mean_and_variance(data, filename):
    #We want to sum over the columns, not the rows. Therefore, we transpose the matrix:
    mean = [sum(values)/len(values) for values in np.transpose(data)]

    mean_str = ' '.join([str(e) for e in mean])
    save_string(mean_str, filename+'-mean.dt')

    diff = [np.subtract(x, mean) for x in data]
    diff_squared = [np.outer(d,np.transpose(d)) for d in diff]
    covariance_ml = reduce(lambda stack, d: np.add(stack, d), diff_squared)

    covariance_ml = np.multiply(covariance_ml, 1.0/len(data))
    cov_str = '\n'.join([' '.join([str(e) for e in row]) for row in covariance_ml])
    save_string(cov_str, filename+'-covariance.dt')

def save_string(str, filename):
    text_file = open(filename, "w")
    text_file.write(str)
    text_file.close()
