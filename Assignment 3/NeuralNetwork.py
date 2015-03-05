__author__ = 'Michael'

import numpy as np

class NeuralNetwork():

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
        self.W = [None]*(len(network_shape)-1)
        for layer in xrange(1,self.number_of_layers):
            self.W[layer-1] = [[1]*(network_shape[layer-1]+1)]*(network_shape[layer])


    def sigmoid(self, x):
        return x/(1.0+x)

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
            self.output[-1][neuron] = self.sigmoid(np.dot(self.output[-2], self.W[-1][neuron]))

        return self.output[-1]


    def calculate_deltas(self, labels):
        #Calculate the deltas in the output layer:
        self.deltas[-1] = np.subtract(self.output[-1], labels).tolist()

        #Go through every layer, calculating deltas:
        for i in range(0,self.number_of_layers-1)[::-1]:
            for neuron in xrange(len(self.output[i])):
                #Calculate derivative of output:
                h_prime = self.d_sigmoid(self.output[i][neuron])

                #Sum for the neurons in next layer:
                sum_next_layer = 0

                neurons_in_next = len(self.output[i+1]) if i == self.number_of_layers-2 else len(self.output[i+1])-1

                for next_neuron in xrange(neurons_in_next):
                    sum_next_layer += self.W[i][next_neuron][neuron]

                #Take the product:
                self.deltas[i][neuron] = h_prime*sum_next_layer


    #Backpropagation algorithm:
    def back_propagation(self, features, labels, training_rate, update=True):
        #Do a forward pass:
        self.forward_pass(features)

        #Computer deltas:
        self.calculate_deltas(labels)

        gradient = self.W[:]

        #Calculate gradient for each edge:
        for layer in xrange(1,self.number_of_layers):
            for neuron in xrange(len(self.W[layer-1])):
                for previous_neuron in xrange(len(self.W[layer-1][neuron])):
                    gradient[layer-1][neuron][previous_neuron] = self.deltas[layer][neuron] * self.output[layer-1][previous_neuron]
                    if update:
                        self.W[layer-1][neuron][previous_neuron] -= training_rate * gradient[layer-1][neuron][previous_neuron]

        return gradient

    def estimate_derivates(self, epsillon):
        gradient = self.W[:]

        #Calculate gradient for each edge:
        for layer in xrange(1,self.number_of_layers):
            for neuron in xrange(len(self.W[layer-1])):
                for previous_neuron in xrange(len(self.W[layer-1][neuron])):
                    gradient[layer-1][neuron][previous_neuron] = error_function(self.W[layer-1][neuron][previous_neuron]+epsillon) - error_function(self.W[layer-1][neuron][previous_neuron]+epsillon)
                    gradient[layer-1][neuron][previous_neuron] /= float(epsillon)

        return gradient

    #Estimate the error of every neuron
    def calculate_error(self):
        pass
'''
Testing Playground:
'''

if __name__ == '__main__':
    nn = NeuralNetwork([2,3,1])

    train = [[0.95,0.05],[0.05,0.95],[0.95,0.95],[0.05,0.05]]
    labels = [[0.95],[0.95],[0.05],[0.05]]

    print nn.back_propagation(train[0],labels[0], 0, update=False)
    #print nn.deltas
    #print nn.estimate_derivates(0.01, lambda x)

    '''
    for i in xrange(10000):
        for j in xrange(4):
            nn.back_propagation(train[j], labels[j],0.9)

    print nn.forward_pass([1,0])
    print nn.forward_pass([0,0])
    print nn.forward_pass([0,1])
    print nn.forward_pass([1,1])
    '''