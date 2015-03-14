__author__ = 'Mia, Daniela and Michael'

'''
Imports:
'''
import NeuralNetwork as NN
import SVM as SVM
import Helper as H
from matplotlib import pyplot as plt

#######
# 1.1 #
#######
nn_2 = NN.NeuralNetwork([1,2,1])
train_data_2,train_labels_2 = nn_2.parse_file('data/sincTrain25.dt')
test_data_2, test_labels_2 = nn_2.parse_file('data/sincValidate10.dt')

nn_20 = NN.NeuralNetwork([1,20,1])
train_data_20,train_labels_20 = nn_20.parse_file('data/sincTrain25.dt')
test_data_20, test_labels_20 = nn_20.parse_file('data/sincValidate10.dt')

print "Gradient (2 neurons): " + str(nn_2.check_gradients(train_data_2, train_labels_2, 10**(-7)))
print "Gradient (20 neurons): " + str(nn_20.check_gradients(train_data_20, train_labels_20, 10**(-7)))

#######
# 1.2 #
#######

nn_2.train_network(train_data_2, train_labels_2, test_data_2, test_labels_2)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
nn_2.plot_samples(ax)
print "Showing plot of estimation of sinc(x) [2 neurons]"
plt.show()
plt.close()


nn_20.train_network(train_data_20, train_labels_20, test_data_20, test_labels_20)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
nn_20.plot_samples(ax)
print "Showing plot of estimation of sinc(x) [20 neurons]"
plt.show()
plt.close()

print "Generating files of learning rate"
for x in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]:
   nn = NN.NeuralNetwork([1,20,1])
   nn.train_network(train_data_20, train_labels_20, test_data_20, test_labels_20, training_rate=x)

#######
# 2.1 #
#######
train_data, train_labels = SVM.parse_file('data/parkinsonsTrainStatML.dt')
test_data, test_labels = SVM.parse_file('data/parkinsonsTestStatML.dt')
norm = H.normalize(train_data)
print "Saving mean and variance of training data in file"
H.save_mean_and_variance(train_data, 'train')
print "Saving mean and variance of normalized training data in file"
H.save_mean_and_variance(norm(train_data), 'norm-train')
print "Saving mean and variance of normalized test data in file"
H.save_mean_and_variance(norm(test_data), 'norm-test')


#######
# 2.2 #
#######

scale = [10**v for v in xrange(-8,8)]
print "Showing plot of crossvalidation error for SVM trained on the raw data"
C,gamma,error = SVM.do_grid_search(train_data, train_labels, scale,scale)
print "Showing plot of crossvalidation error for SVM trained on the normalized training data"
norm_C, norm_gamma, norm_error = SVM.do_grid_search(norm(train_data), train_labels, scale,scale)

print "Raw data:", C, gamma, error
print "Normalized data:", norm_C, norm_gamma, norm_error

clf_raw = SVM.create_and_fit(train_data, train_labels, C, gamma)
print "Raw training error:", SVM.evaluate(clf_raw, train_data, train_labels)
print "Raw test error:", SVM.evaluate(clf_raw, test_data, test_labels)

clf_norm = SVM.create_and_fit(norm(train_data), train_labels, norm_C, norm_gamma)
print "Normalized training error:", SVM.evaluate(clf_norm, norm(train_data), train_labels)
print "Normalized test error:", SVM.evaluate(clf_norm, norm(test_data), test_labels)

#########
# 2.3.1 #
#########
print "Numbers of bounded and free support vectors is being written to text file..."
count_support_vectors(norm(train_data), train_labels, scale, norm_gamma)