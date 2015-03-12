__author__ = 'Daniela, Mia, and Michael'

'''
Imports:
'''
import codecs
import sys
import numpy as np
from itertools import chain
from sklearn import svm
import Preprocessing as n

'''
I/O:
'''
def parse_file(file_name):
        print >>sys.stderr, "Reading from file \'"+file_name+"\'..."
        features = []
        targets = []
        for line in codecs.open(file_name, encoding='utf-8'):
            if line != '\n':
                feature = line.strip().split(' ')
                features.append([float(f) for f in feature[:-1]])
                targets.append(int(feature[-1]))

        return features, targets

'''
Training:
'''

def create_and_fit(train_data, train_labels, C, gamma):
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(train_data, train_labels)
    return clf

'''
Evaluation:
'''

def evaluate(clf, validate_data, validate_labels):
    predictions = clf.predict(validate_data)
    true_positives = np.dot(predictions, validate_labels)
    return true_positives/float(len(validate_data))

'''
Cross-validation:
'''
####################################
# Performs N-fold cross-validation #
####################################
def cross_validate(data, labels, k, params):
    #Split into evenly sized chunks
    samples_per_fold = len(data)/k
    data_folds = [list(t) for t in zip(*[iter(data)]*samples_per_fold)]
    label_folds = [list(t) for t in zip(*[iter(labels)]*samples_per_fold)]

    #Distribute the remainder evenly over the folds
    leftover_data = data[samples_per_fold*k:]
    leftover_labels = labels[samples_per_fold*k:]
    for i in xrange(len(leftover_data)):
        data_folds[i%k].append((leftover_data[i]))
        label_folds[i%k].append((leftover_labels[i]))

    acc = 0

    #Do the experiments
    for i in xrange(len(data_folds)):
        #Get a view of the data
        train_data = data_folds[:]
        train_labels = label_folds[:]

        #Construct training and test sets
        validate_data = train_data.pop(i)
        validate_labels = train_labels.pop(i)
        train_data = list(chain.from_iterable(train_data))
        train_labels = list(chain.from_iterable(train_labels))

        #Create a classifier and fit it to the data:
        cvf = create_and_fit(train_data, train_labels, *params)


        #Evaluate accuracy
        acc += evaluate(cvf, validate_data, validate_labels)

    return acc/float(k)


train_data, train_labels = parse_file('data/parkinsonsTrainStatML.dt')
test_data, test_labels = parse_file('data/parkinsonsTestStatML.dt')

norm = n.normalize(train_data)

print cross_validate(norm(train_data), train_labels, 5, [10000, 0.001])


#clf = svm.SVC(kernel='rbf')
#clf.fit(train_data, train_labels)
#print evaluate(clf, test_data, test_labels)

# get support vectors
#print clf.support_vectors_
# get indices of support vectors
#print clf.support_
# get number of support vectors for each class
#print clf.n_support_
