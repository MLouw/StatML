__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import sys
import codecs
import numpy as np
import Utility

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
            
    def evaluate(self, test_features, test_labels):
        error = 0
        for i in xrange(len(test_features)):
            if self.predict(test_features[i]) != test_labels[i]:
                error += 1

        return float(error)/len(test_features)



'''Test playground'''
lc = Linear_Classifier()

train_data, train_labels = lc.from_file('IrisTrain2014.dt')
test_data, test_labels = lc.from_file('IrisTest2014.dt')
n_function = Utility.normalize(train_data)
normalized_train_data = n_function(train_data)
normalized_test_data = n_function(test_data)



lc.fit(normalized_train_data, train_labels)
print lc.evaluate(normalized_train_data, train_labels)
print lc.evaluate(normalized_test_data, test_labels)


