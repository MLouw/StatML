__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import linear_analysis as la
import Utility
from matplotlib import pyplot as plt


lc = la.Linear_Classifier()

train_data, train_labels = lc.from_file('IrisTrain2014.dt')
test_data, test_labels = lc.from_file('IrisTest2014.dt')
n_function = Utility.normalize(train_data)
normalized_train_data = n_function(train_data)
normalized_test_data = n_function(test_data)


lc.fit(normalized_train_data, train_labels)
print lc.evaluate(normalized_train_data, train_labels)
print lc.evaluate(normalized_test_data, test_labels)

#TODO find out what the hell is wrong with the plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Utility.plot_irises(train_data, train_labels, ax)

ax.plot([0,lc.W[0][2]*20.0], [lc.W[0][0], -lc.W[0][1]*20.0+lc.W[0][0]])

print -lc.W[0][2]*20.0, lc.W[0][1]*20.0
print lc.W

plt.show()

