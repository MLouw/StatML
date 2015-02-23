__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import sys
import codecs
import numpy as np
import Utility
import math

##
# Linear classifier used in exercise 1.1 and 1.2
#
class Linear_Classifier():

    def __init__(self):
        self.number_of_classes = 0
        self.number_of_features = 0
        self.W = None

    def from_file(self, file_name):
        print >>sys.stderr, "Reading from file \'"+file_name+"\'..."
        features = []
        labels = []
        for line in codecs.open(file_name, encoding='utf-8'):

            feature = line.strip().split(' ')
            features.append([float(f) for f in feature[:-1]])
            labels.append(int(feature[-1]))

        return features, labels

# train the model to fit the data given in features and labels
    def fit(self, features, labels):
        self.number_of_classes = len(set(labels))
        self.number_of_features = len(features[0])
        # add in the bias
        X = [np.concatenate(([1],f)) for f in features]  
        #calculate T as given by BISHOP
        T = []

        for l in labels: 
            t = [0]*self.number_of_classes
            t[l] = 1
            T.append(t)

        #calculate pseudo inverse with the .pinv        
        X_pinverse = np.linalg.pinv(X)
        #calculate W as specified by BISHOP
        self.W = np.dot(X_pinverse, T) 
        #transposed for easier lookup
        self.W = np.transpose(self.W)

        print self.W

    def fit_fishers(self, features, labels):
        self.number_of_classes = len(set(labels))
        self.number_of_features = len(features[0])

        #Divide the features into classes:
        classes = [[features[p] for p in xrange(len(features)) if labels[p]==i] for i in xrange(self.number_of_classes)]

        #Calculate the mean of every class:
        class_means = [Utility.mean(data) for data in classes]

        #Estimate a common covariance:
        covariance = [[0,0],[0,0]]
        for i in xrange(self.number_of_classes):
            covariance += Utility.covariance(classes[i], class_means[i])
        covariance = np.multiply(covariance, 1.0/(len(data)-self.number_of_classes))

        #Calculate class probabilities:
        class_log_probabilities = [math.log(len(c)/float(len(features))) for c in classes]

        #Construct the W:
        self.W = [None]*self.number_of_classes
        for i in xrange(self.number_of_classes):
            #Initialize the row:
            self.W[i] = [None]*(self.number_of_features+1)

            #Build the constant term:
            self.W[i][0] = class_log_probabilities[i] - 0.5*np.dot(class_means[i], np.dot(np.linalg.inv(covariance), class_means[i]))

            #Build the x coefficients:
            x_coeff = np.dot(np.linalg.inv(covariance), class_means[i])
            self.W[i][1:] = x_coeff

            #Turn the list into a numpy array for easier manipulaion:
            self.W[i] = np.array(self.W[i])

        print self.W

    def predict(self, feature):
        x = np.concatenate(([1],feature))
        best_i = None
        best_y_class = float("-Inf")
        for i in xrange(self.number_of_classes):
            y_class = np.dot(x, self.W[i])
            if y_class > best_y_class:
                best_y_class = y_class
                best_i = i

        return best_i

    def predict_all(self, features):
        return [self.predict(feature) for feature in features]

    def evaluate(self, test_features, test_labels):
        error = 0
        for i in xrange(len(test_features)):
            if self.predict(test_features[i]) != test_labels[i]:
                error += 1

        return float(error)/len(test_features)

    def draw_discrimination_functions(self, ax, colors=[[1,0,0],[0,1,0],[0,0,1]], iterations=5):
        for i in xrange(self.number_of_classes):
            xd = -self.W[i][0]/self.W[i][1]
            yd = self.W[i][0]/self.W[i][2]

            ax.plot([-xd*iterations, xd*iterations], [-yd*(iterations+1),  yd*(iterations-1)], color=colors[i])

    def draw_decision_boundaries(self, ax, iterations=20):
        ls = [None]*3
        ls[0] = self.W[0] - self.W[1]
        ls[1] = self.W[1] - self.W[2]
        ls[2] = self.W[2] - self.W[0]

        px, py = Utility.line_intersection(ls[0], ls[1])

        for i in xrange(3):
            xd = -ls[i][0]/ls[i][1]
            yd = ls[i][0]/ls[i][2]

            limx = xd*iterations
            limy = yd*(iterations-1)

            if np.dot([1,limx, limy], self.W[(i+2)%3]) > 0:
                limx = -limx
                limy = -limy

            ax.plot([px, limx], [py,  limy],color=[0,0,0])

'''Test playground'''
if __name__ == '__main__':
    lc = Linear_Classifier()

    train_data, train_labels = lc.from_file('IrisTrain2014.dt')
    test_data, test_labels = lc.from_file('IrisTest2014.dt')
    n_function = Utility.normalize(train_data)
    normalized_train_data = n_function(train_data)
    normalized_test_data = n_function(test_data)



    lc.fit(normalized_train_data, train_labels)
    print lc.evaluate(normalized_train_data, train_labels)
    print lc.evaluate(normalized_test_data, test_labels)


