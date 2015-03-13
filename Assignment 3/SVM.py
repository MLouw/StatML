__author__ = 'Daniela, Mia, and Michael'

'''
Imports:
'''
import codecs
import sys
import numpy as np
from itertools import chain
from sklearn import svm
import Helper as n
from matplotlib import pyplot as plt

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
    errors = np.sum(np.logical_xor(predictions, validate_labels))
    return errors/float(len(validate_data))

def do_grid_search(data, labels, C_range, gamma_range):
    results = np.ndarray((len(C_range),len(gamma_range)))

    best = None
    best_result = 1.0
    for i,C in enumerate(C_range):
        for j,gamma in enumerate(gamma_range):
            results[i][j] = cross_validate(data, labels, 5, [C, gamma])

            if results[i][j] < best_result:
                best_result = results[i][j]
                best = (C, gamma)

    # draw heatmap of accuracy as a function of gamma and C
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.95)
    plt.imshow(results, interpolation='nearest', cmap=plt.cm.spectral)
    plt.clim(0,1)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)

    plt.show()

    return best[0], best[1], best_result

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


def count_support_vectors(data, labels, C_range, gamma):
    bound = []
    free = []
    for C in C_range:
        clf = create_and_fit(data, labels, C, gamma)

        coefficients = clf.dual_coef_[0]
        bounded = [c for c in coefficients if c == C]


        bound.append(str(len(bounded)))
        free.append(str(len(coefficients) - len(bounded)))


    result = zip([str(thingy) for thingy in C_range], bound, free)

    n.save_string('\n'.join([' '.join(elem) for elem in result]), 'bound-and-free-support-vectors.dt')


train_data, train_labels = parse_file('data/parkinsonsTrainStatML.dt')
test_data, test_labels = parse_file('data/parkinsonsTestStatML.dt')

norm = n.normalize(train_data)
n.save_mean_and_variance(train_data, 'train')
n.save_mean_and_variance(norm(train_data), 'norm-train')
n.save_mean_and_variance(norm(test_data), 'norm-test')

scale = [10**v for v in xrange(-8,8)]
C,gamma,error = do_grid_search(train_data, train_labels, scale,scale)
norm_C, norm_gamma, norm_error = do_grid_search(norm(train_data), train_labels, scale,scale)

print "Raw data:", C, gamma, error
print "Normalized data:", norm_C, norm_gamma, norm_error

clf_raw = create_and_fit(train_data, train_labels, C, gamma)
print "Raw training error:", evaluate(clf_raw, train_data, train_labels)
print "Raw test error:", evaluate(clf_raw, test_data, test_labels)

clf_norm = create_and_fit(norm(train_data), train_labels, norm_C, norm_gamma)
print "Normalized training error:", evaluate(clf_norm, norm(train_data), train_labels)
print "Normalized test error:", evaluate(clf_norm, norm(test_data), test_labels)

#clf = svm.SVC(kernel='rbf')
#clf.fit(train_data, train_labels)
#print evaluate(clf, test_data, test_labels)

count_support_vectors(norm(train_data), train_labels, scale, norm_gamma)

'''
coefficients = clf_norm.dual_coef_[0]
bounded = [c for c in coefficients if c == norm_C]

print "Bounded support vectors:", len(bounded)
print "Free support vectors:", len(coefficients) - len(bounded)
'''

# get support vectors
#print clf.support_vectors_
# get indices of support vectors
#print clf.support_
# get number of support vectors for each class
#print clf.n_support_
