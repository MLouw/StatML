__author__ = 'Michael'

import numpy as np
import sys
import codecs
import math
from matplotlib import pyplot as plt
import Helper

class NeuralNetwork():

    #Fields:
    #Todo

    #Constructor:
    def __init__(self, network_shape):
        #Counts the number of layers:
        self.number_of_layers = len(network_shape)

        #Initiates the output and deltas to an array of same shape as the network:
        self.output = [None]*len(network_shape)
        self.deltas = [None]*len(network_shape)
        for layer in xrange(self.number_of_layers-1):
            self.output[layer] = [None]*(network_shape[layer]+1)
            self.deltas[layer] = [None]*(network_shape[layer]+1)
        self.output[self.number_of_layers-1] = [None]*network_shape[-1]
        self.deltas[self.number_of_layers-1] = [None]*network_shape[-1]

        #Initialize the weights so that W[l][i][j] is the weights from neuron j in layer l to neuron i:
        self.W = np.array([None]*(len(network_shape)-1))
        for layer in xrange(1,self.number_of_layers):
            self.W[layer-1] = np.array([np.array([0.5]*(network_shape[layer-1]+1))]*(network_shape[layer]))
            self.W[layer-1] = 0.2 * np.random.random_sample((network_shape[layer],network_shape[layer-1]+1)) - 0.1

    def sigmoid(self, x):
        return x/(1.0+abs(x))

    def d_sigmoid(self, x):
        return 1.0/((1.0+abs(x))**2)

    #Computes a forward pass through the network:
    def forward_pass(self, features):
        #The output of the first layer is just the features:
        self.output[0][:-1] = features
        self.output[0][-1] = 1

        #Compute the output of a layer:
        for layer in xrange(1, self.number_of_layers-1):
            #Compute the output of a single neuron as in the perceptron algorithm:
            for neuron in xrange(len(self.output[layer])-1):
                self.output[layer][neuron] = self.sigmoid(np.dot(self.output[layer-1], self.W[layer-1][neuron]))
            self.output[layer][-1] = 1

        #calculate the output layer:
        for neuron in xrange(len(self.output[-1])):
            self.output[-1][neuron] = np.dot(self.output[-2], self.W[-1][neuron])

        return self.output[-1]


    def calculate_deltas(self, labels):
        #Calculate the deltas in the output layer:
        self.deltas[-1] = np.subtract(self.output[-1], labels)

        error = 0.5*sum([self.deltas[-1][i]**2 for i in xrange(len(self.deltas[-1]))])

        #Go through every layer in the reverse order, calculating deltas:
        for i in range(1,self.number_of_layers-1)[::-1]:
            for neuron in xrange(len(self.output[i])-1):
                #Calculate derivative of output:
                h_prime = self.d_sigmoid(np.dot(self.output[i-1], self.W[i-1][neuron]))

                #Sum for the neurons in next layer:
                sum_next_layer = 0

                neurons_in_next = len(self.output[i+1]) if i == self.number_of_layers-2 else len(self.output[i+1])-1

                for next_neuron in xrange(neurons_in_next):
                    sum_next_layer += self.W[i][next_neuron][neuron]*self.deltas[i+1][next_neuron]

                #Take the product:
                self.deltas[i][neuron] = h_prime*sum_next_layer

        return error

    def train_network(self, train_features, train_labels, test_features, test_labels, training_rate=0.2, stopping_value=0.0005, max_iterations=12000):
        itera = ['Epoch']
        itera.extend([str(nu*500) for nu in xrange(max_iterations/500)])
        train_eval = ['Train']
        test_eval = ['Test']
        for j in xrange(max_iterations):
            update = nn.train_epoch(train_features, train_labels, training_rate=training_rate)
            s_u = 0
            for i in xrange(len(update)):
                s_u += np.sum(np.multiply(update[i],update[i]))
            update_norm = math.sqrt(s_u)

            if j%500 == 499:
                train_eval.append(str(nn.evaluate(train_features, train_labels)))
                test_eval.append(str(nn.evaluate(test_features, test_labels)))

            if update_norm < stopping_value:
                train_eval.append(str(nn.evaluate(train_features, train_labels)))
                test_eval.append(str(nn.evaluate(test_features, test_labels)))

        #Fill arrays if the aren't full:
        while(len(train_eval) < len(itera)):
            train_eval.append(str(train_eval[-1]))
            test_eval.append(str(test_eval[-1]))

        results = zip(itera, train_eval, test_eval)

        Helper.save_string('\n'.join([' '.join(elem) for elem in results]), 'nn-'+str(len(self.output[1]))+'-error-learning-rate-'+str(training_rate)+'.dt')

    def train_epoch(self, features, labels, training_rate=0.2):
        gradient_sum = np.array([np.array([np.zeros(len(neuron)) for neuron in layer]) for layer in self.W])

        for i in xrange(len(features)):
            _, gradient = self.back_propagation(features[i], labels[i], training_rate=training_rate, update=False)

            for i in xrange(len(gradient_sum)):
                gradient_sum[i] = np.add(gradient_sum[i], gradient[i])

        weighted_gradients = np.multiply(-training_rate*1.0/len(features), gradient_sum)
        self.W = np.add(weighted_gradients, self.W)

        return weighted_gradients

    #Backpropagation algorithm:
    def back_propagation(self, features, labels, training_rate=0.2, update=True):
        #Do a forward pass:
        self.forward_pass(features)

        #Compute deltas:
        error = self.calculate_deltas(labels)

        gradients = [[[None]*len(neuron) for neuron in layer] for layer in self.W]

        #Calculate gradient for each edge:
        for layer in xrange(1,self.number_of_layers):
            for neuron in xrange(len(self.W[layer-1])):
                for previous_neuron in xrange(len(self.W[layer-1][neuron])):
                    temp = self.deltas[layer][neuron] * self.output[layer-1][previous_neuron]
                    gradients[layer-1][neuron][previous_neuron] = temp
                    if update:
                        self.W[layer-1][neuron][previous_neuron] -= training_rate * temp

        return error,gradients

    def evaluate(self, features, labels):
        error_sum = 0
        for i in xrange(len(features)):
            out =self.forward_pass(features[i])

            errors = np.subtract(out, labels[i])
            error_sum += 0.5*sum([e**2 for e in errors])

        return error_sum / float(len(features))


    def check_gradients(self, features, labels, epsillon):
        s = 0

        #Run through all the samples:
        for i in xrange(len(features)):
            #Calculate the gradient with backprop:
            error,gradients = self.back_propagation(features[i], labels[i], update=False)

            #Initialize gradient:
            fd_gradients = [[[None]*len(neuron) for neuron in layer] for layer in self.W]

            #Calculate the gradient with finite differences:
            for layer in xrange(1,self.number_of_layers):
                for neuron in xrange(len(self.W[layer-1])):
                    for previous_neuron in xrange(len(self.W[layer-1][neuron])):
                        # Update weight:
                        self.W[layer-1][neuron][previous_neuron] += epsillon

                        #Do a forward pass:
                        self.forward_pass(features[i])

                        #Compute updated error:
                        updated_errors = np.subtract(self.output[-1], labels[i])
                        new_error = 0.5*sum([updated_errors[j]**2 for j in xrange(len(updated_errors))])

                        #Set gradient:
                        fd_gradients[layer-1][neuron][previous_neuron] = (new_error - error)/float(epsillon)

                        #Reset weight:
                        self.W[layer-1][neuron][previous_neuron] -= epsillon

            s += self.gradient_sum_of_squares(gradients, fd_gradients)

        return s/float(len(features))

    #Calculate sum of squares error for two network gradients:
    def gradient_sum_of_squares(self, gradient1, gradient2):
        s=0
        for layer in xrange(len(self.W)):
            for i in xrange(len(self.W[layer])):
                for j in xrange(len(self.W[layer][i])):
                    s+=(gradient1[layer][i][j]-gradient2[layer][i][j])**2

        return s


    # Parses a data file:
    def parse_file(self, file_name):
        print >>sys.stderr, "Reading from file \'"+file_name+"\'..."
        features = []
        targets = []
        for line in codecs.open(file_name, encoding='utf-8'):
            if line != '\n':
                feature = line.strip().split(' ')
                features.append([float(f) for f in feature[:-1]])
                targets.append(float(feature[-1]))

        return features, targets

    # Plots a series of samples from the learned function:
    def plot_samples(self,ax):

        xs = [float(j)/100 for j in xrange(-1000, 1001, 5)]
        predicted_ys = [self.forward_pass([x])[0] for x in xs]

        should_have_been_ys = [math.sin(x)/x for x in xs if x!=0]

        x_offset = 0.5
        y_offset = 0.3

        ax.set_xlim([xs[0], xs[-1]])
        ax.set_ylim([min(predicted_ys)-y_offset, max(predicted_ys)+y_offset])

        ax.set_xlabel('x')
        ax.set_ylabel('sinc(x)')

        ax.set_xticks(np.arange(xs[0], xs[-1]+0.0001,2.0))
        ax.set_yticks(np.arange(-0.5,1.5,0.5))


        ax.plot(xs,predicted_ys, color=[1,0,0], label="Our estimation")
        ax.plot([x for x in xs if x!=0],should_have_been_ys, color=[0,1,0], label="Actual value")

        plt.gca().legend(loc='upper right')
        plt.grid()



'''
Testing Playground:
'''

if __name__ == '__main__':
    nn = NeuralNetwork([1,2,1])

    train_data,train_labels = nn.parse_file('data/sincTrain25.dt')
    test_data, test_labels = nn.parse_file('data/sincValidate10.dt')

    print nn.check_gradients(train_data, train_labels, 10**(-7))

    #nn.train_epoch(train_data, train_labels)
    #print nn.evaluate(test_data, test_labels)

    nn.train_network(train_data, train_labels, test_data, test_labels)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    nn.plot_samples(ax)
    plt.show()
    plt.close()

    nn = NeuralNetwork([1,20,1])

    train_data,train_labels = nn.parse_file('data/sincTrain25.dt')
    test_data, test_labels = nn.parse_file('data/sincValidate10.dt')

    print nn.check_gradients(train_data, train_labels, 10**(-7))

    #nn.train_epoch(train_data, train_labels)
    #print nn.evaluate(test_data, test_labels)

    nn.train_network(train_data, train_labels, test_data, test_labels)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    nn.plot_samples(ax)
    plt.show()
    plt.close()


