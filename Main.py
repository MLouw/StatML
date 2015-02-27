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


#######
# 1.1 #
#######

lc.fit_fishers(train_data, train_labels)
print "Training error for LDA: " + str(lc.evaluate(train_data, train_labels))
print "Test error for LDA: " + str(lc.evaluate(test_data, test_labels))

print "Showing plot of the data"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(train_data, lc.predict_all(train_data), ax)
plt.show()
plt.close()

print "Showing plot of the data with discrimination functions"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(train_data, lc.predict_all(train_data), ax, title="with discrimination functions")
lc.draw_discrimination_functions(ax)
plt.show()
plt.close()

print "Showing plot of the data with discrimination functions and decision boundaries"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(train_data, lc.predict_all(train_data), ax, title="with discrimination functions and decision boundaries")
lc.draw_discrimination_functions(ax)
lc.draw_decision_boundaries(ax)
plt.show()
plt.close()

print "Showing plot of the test data with discrimination functions and decision boundaries"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(test_data, lc.predict_all(test_data), ax, title="for the test data with discrimination functions and decision boundaries")
lc.draw_discrimination_functions(ax)
lc.draw_decision_boundaries(ax)
plt.show()
plt.close()

#######
# 1.2 #
#######
lc.fit_fishers(normalized_train_data, train_labels)
print "Training error on the transformed data: " + str(lc.evaluate(normalized_train_data, train_labels))
print "Test error on the transformed data: " + str(lc.evaluate(normalized_test_data, test_labels))

#######
# 2.1 #
#######
lr = R.LinearRegression()
train_data, train_targets = lr.parse_file("sunspotsTrainStatML.dt")
test_data, test_targets = lr.parse_file("sunspotsTestStatML.dt")

train_limit_2 = lr.limit_columns([4], train_data)
test_limit_2 = lr.limit_columns([4], test_data)
train_limit_1 = lr.limit_columns([2,3], train_data)
test_limit_1 = lr.limit_columns([2,3], test_data)

lr.fit(train_limit_2, train_targets)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
test_prediction = lr.predict_all(test_limit_2)
lr.draw_datapoints(ax, train_limit_2, train_targets, point_color=[1,0,0], label='Training data')
lr.draw_datapoints(ax, test_limit_2, test_targets, point_color=[0,1,0], label='Test targets')
lr.draw_datapoints(ax, test_limit_2, test_prediction, point_color=[0,0,1], label='Test predictions')

print "Showing plot of x and y variables of training set"
ax.set_title('Sunspots in year t versus year t-16')
ax.set_xlabel('Sunspots in year t-16')
ax.set_ylabel('Sunspots in year t')
plt.show()
plt.close()

print "Showing plot of predicted sunspots vs. years"
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

lr.plot_evaluation(ax, 1861, test_targets, label='Actual', color=[0,1,0])
lr.plot_evaluation(ax, 1861, test_prediction, label='Model 2')
print "Model 2:",lr.evaluate(test_limit_2, test_targets)
lr.fit(train_data, train_targets)
test_prediction2 = lr.predict_all(test_data)
print "Model 3:",lr.evaluate(test_data, test_targets)
lr.plot_evaluation(ax, 1861, test_prediction2, label='Model 3', color=[0,0,1])
lr.fit(train_limit_1, train_targets)
test_prediction3 = lr.predict_all(test_limit_1)
print "Model 1:",lr.evaluate(test_limit_1, test_targets)
lr.plot_evaluation(ax, 1861, test_prediction3, label='Model 1', color=[0,1,1])

ax.set_title('Predicted sunspots vs. years')
plt.show()
plt.close()

#######
# 2.2 #
#######

print "Showing plot of RMS error for different values of the prior precision parameter alpha"
lr = R.LinearRegression()
lr.fit(train_limit_2, train_targets, method='MAP', alpha=0.1)
print lr.evaluate(test_limit_2, test_targets)

R.compare_methods(train_data, train_targets, test_data, test_targets)
plt.close()

m1 = [train_limit_1, train_targets, test_limit_1, test_targets, '-r', 'Model 1']
m2 = [train_limit_2, train_targets, test_limit_2, test_targets, '-g', 'Model 2']
m3 = [train_data, train_targets, test_data, test_targets, '-b', 'Model 3']

R.compare_models_for_alpha([m1, m2, m3])
plt.close()
