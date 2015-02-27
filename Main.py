__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import linear_analysis as la
import Utility
import Regression as R
from matplotlib import pyplot as plt

lc = la.Linear_Classifier()

train_data, train_labels = lc.from_file('IrisTrain2014.dt')
test_data, test_labels = lc.from_file('IrisTest2014.dt')
n_function = Utility.normalize(train_data)
normalized_train_data = n_function(train_data)
normalized_test_data = n_function(test_data)
lc.fit_fishers(normalized_train_data, train_labels)

#######
# 1.1 #
#######

print "Training error for LDA: " + str(lc.evaluate(normalized_train_data, train_labels))
print "Test error for LDA: " + str(lc.evaluate(normalized_test_data, test_labels))

print "Showing plot of the data"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(normalized_train_data, lc.predict_all(normalized_train_data), ax)
plt.show()

print "Showing plot of the data with discrimination functions"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(normalized_train_data, lc.predict_all(normalized_train_data), ax, title="with discrimination functions")
lc.draw_discrimination_functions(ax)
plt.show()

print "Showing plot of the data with discrimination functions and decision boundaries"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(normalized_train_data, lc.predict_all(normalized_train_data), ax, title="with discrimination functions and decision boundaries")
lc.draw_discrimination_functions(ax)
lc.draw_decision_boundaries(ax)
plt.show()

print "Showing plot of the test data with discrimination functions and decision boundaries"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(normalized_test_data, lc.predict_all(normalized_test_data), ax, title="for the test data with discrimination functions and decision boundaries")
lc.draw_discrimination_functions(ax)
lc.draw_decision_boundaries(ax)
plt.show()

#######
# 1.2 #
#######

print "Training error on the transformed data: "
print "Test error on the transformed data: "

#######
# 2.1 #
#######

print "Showing plot of x and y variables of training set"
print "Showing plot of years vs. predicted sunspots"

#######
# 2.2 #
#######

print "Showing plot of RMS error for different values of the prior precision parameter alpha"
