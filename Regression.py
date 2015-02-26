__author__ = 'Mia, Daniela, Michael'

'''
Imports:
'''
import sys
import codecs
import numpy as np
from matplotlib import pyplot as plt

'''
Linear regression class:
'''
class LinearRegression():

    def __init__(self):
        self.w = None

    '''
    I/O:
    '''

    #Parses a data file:
    def parse_file(self, file_name):
        print >>sys.stderr, "Reading from file \'"+file_name+"\'..."
        features = []
        targets = []
        for line in codecs.open(file_name, encoding='utf-8'):

            feature = line.strip().split(' ')
            features.append([float(f) for f in feature[:-1]])
            targets.append(float(feature[-1]))

        return features, targets

    #Fit the model to a set of features and labels given in a specific file:
    def fit_to_file(self, filename, method='ML', alpha=None):
        features, targets = self.parse_file(filename)
        self.fit(features, targets, method=method, alpha=alpha)

    #Draws a set of data points:
    def draw_datapoints(self, ax, xs, ys, point_color=[0,0,0], label='<No label>'):
        ax.scatter(xs,ys, color=point_color, label=label)
        plt.gca().legend(loc='upper right')
    '''
    Metrics & Constants:
    '''

    #Applies the basis functions to a vector of features. An additional bias function is added:
    def apply_basis_functions(self, x):
        return np.concatenate(([1], x))

    '''
    Training:
    '''

    #Limit a feature matrix to the specified columns:
    def limit_columns(self, used_columns, features):
        return [[feature[i] for i in used_columns] for feature in features]

    #Fit the model to a set of features and labels using the specified method:
    def fit(self, features, targets, method='ML', alpha=None):
        if method == 'ML':
            return self.fit_maximum_likelihood(features, targets)
        elif method == 'MAP':
            return self.fit_maximum_a_posteori(features, targets, alpha)
        else:
            print>>sys.stderr, "The method name "+method+" is not valid."


    #Fit the model a set of features and labels using the maximum likelihood method:
    def fit_maximum_likelihood(self, features, targets):
        #Calculate the design matrix as specified by Bishop:
        design_matrix = [self.apply_basis_functions(x) for x in features]

        #Take the pseudoinverse:
        pinv_design_matrix = np.linalg.pinv(design_matrix)

        #Calculate the weight vector:
        self.w = np.dot(pinv_design_matrix, targets)

        print>>sys.stderr, "The model was fitted using maximum likelihood."

    #Fit the model to a set of features and labels using the a posteori method:
    def fit_maximum_a_posteori(self, features, targets, alpha, beta=1):
        #Estimate the covariance:
        covariance = self.posterior_covariance(features, alpha, beta)

        #Estimate the mean:
        mean = self.posterior_mean(covariance, features, targets, beta)

        self.w = mean
        print>>sys.stderr, "The model was fitted using maximum a posteori."

    #Estimate the posterior mean:
    def posterior_mean(self, posterior_covariance, features, targets, beta=1):
        #Calculate the design matrix as specified by Bishop:
        design_matrix = [self.apply_basis_functions(x) for x in features]

        #Follow the calculation given by Bishop:
        return np.multiply(beta, np.dot(posterior_covariance, np.dot(np.transpose(design_matrix), targets)))

    #Estimate the posterior covariance:
    def posterior_covariance(self, features, alpha, beta=1):
        #Calculate the design matrix as specified by Bishop:
        design_matrix = [self.apply_basis_functions(x) for x in features]

        #Calculate the second term:
        st = np.multiply(beta, np.dot(np.transpose(design_matrix), design_matrix))

        #Calculate the first term:
        ft = np.multiply(alpha, np.ones_like(st))

        #Return the inverse of the sum:
        return np.linalg.inv(np.add(ft, st))
    '''
    Prediction:
    '''

    #Return the prediction for the given feature:
    def predict(self, feature):
        return np.dot(self.w, self.apply_basis_functions(feature))

    #Return a list of predictions corresponding to a list of features:
    def predict_all(self, features):
        l = [None]*len(features)

        for i in xrange(len(features)):
            l[i] = self.predict(features[i])

        return l

    '''
    Evaluation:
    '''

    #Evaluate the performance of the regression through mean square error:
    def evaluate(self, features, targets):
        #Make the predictions:
        predictions = self.predict_all(features)

        #Calculate square differences:
        s_diff = [(targets[i]-predictions[i])**2 for i in xrange(len(predictions))]

        #Normalize and take the square root:
        return np.sqrt(1.0/len(predictions)*sum(s_diff))

    #Prints a time series of year vs. prediction:
    def print_evaluation(self, targets):
        for i in xrange(len(targets)):
            print 1961+i, targets[i]

    #Performs a comparison of the maximum likelihood and maximum a posteori methods:
    def compare_methods(self, training_data, training_targets, test_data, test_targets):
        pass

'''
Testing playground:
'''
if __name__ == '__main__':
    print "Doing maximum likelihood..."
    lr = LinearRegression()
    train_data, train_targets = lr.parse_file("sunspotsTrainStatML.dt")
    test_data, test_targets = lr.parse_file("sunspotsTestStatML.dt")

    train_limit_2 = lr.limit_columns([4], train_data)
    test_limit_2 = lr.limit_columns([4], test_data)

    lr.fit(train_limit_2, train_targets)
    print lr.evaluate(test_limit_2, test_targets)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    test_prediction = lr.predict_all(test_limit_2)

    lr.draw_datapoints(ax, train_limit_2, train_targets, point_color=[1,0,0], label='Training data')
    lr.draw_datapoints(ax, test_limit_2, test_targets, point_color=[0,1,0], label='Test targets')
    lr.draw_datapoints(ax, test_limit_2, test_prediction, point_color=[0,0,1], label='Test predictions')

    plt.show()

    print "Doing maximum a posteori..."
    lr = LinearRegression()
    lr.fit(train_limit_2, train_targets, method='MAP', alpha=0.1)
    print lr.evaluate(test_limit_2, test_targets)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    test_prediction = lr.predict_all(test_limit_2)

    lr.draw_datapoints(ax, train_limit_2, train_targets, point_color=[1,0,0], label='Training data')
    lr.draw_datapoints(ax, test_limit_2, test_targets, point_color=[0,1,0], label='Test targets')
    lr.draw_datapoints(ax, test_limit_2, test_prediction, point_color=[0,0,1], label='Test predictions')

    plt.show()

