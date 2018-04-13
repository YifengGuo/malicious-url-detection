import numpy as np
from functools import reduce
import time

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LogitRegression(object):

    def __init__(self, input_num):
        # self.weights = [0.0 for _ in range(input_num)]  # by default each weight for features is set as 1
        self.weights = np.ones((input_num, 1))
        self.bias = 0.0
    #
    # def train_logit_regression(self, X_train, labels_train, iterations, ita):
    #     '''
    #
    #     :param X_train: train data set
    #     :param labels_train: train labels
    #     :param iterations: number of iteration
    #     :param ita: learning rate
    #     :return: weights calculated by gradient descent
    #     '''
    #
    #     # sampleNum, featureNum = np.shape(X_train)  # row number of train dataset is sample number, column number is features
    #     featureNum = len(X_train)
    #     weights = np.ones((featureNum, 1))  # set default weights as 1
    #
    #     output = sigmoid(np.dot(X_train, weights))
    #     # print(type(output[0]))
    #     error = float(labels_train) - float(output[0])
    #     print(type(error))
    #     weights = weights + ita * X_train.T * error
    #     print(weights)
    #
    #     return weights
    #
    # def test_accuarcy(self, X_test, labels_test, weights):
    #     numSamples, numFeatures = np.shape(X_test)
    #     matchCount = 0
    #     for i in range(numSamples):
    #         predict = sigmoid(X_test[i, :] * weights)[0, 0] > 0.5
    #         if predict == bool(labels_test[i, 0]):
    #             matchCount += 1
    #     accuracy = float(matchCount) / numSamples
    #     return accuracy

    def calc_output(self, input_vec):
        '''
        calculate sigmoid(w * X)
        :param input_vec: x1 x2 ....xm
        :return: 
        '''
        # return sigmoid(
        #     reduce(
        #         lambda a, b: a + b,
        #         map(lambda x_w: x_w[0] * x_w[1],
        #             zip(input_vec, self.weights)), 0.0
        #     ) + self.bias
        # )

        return sigmoid(np.dot(input_vec, self.weights))

    def train(self, input_vecs, labels, iteration, rate):
        '''
        train all the dataset given iteration times and learning rate
        :param input_vecs: tf-idf weighted document matrix   shape(urls * features)
        :param labels: labels 
        :param iteration: iteration times
        :param rate: learning rate
        :return: 
        '''
        for i in range(iteration):
            # print('iteration: %d ' %i)
            self.train_one_iteration(input_vecs, labels, rate)

    def train_one_iteration(self, input_vecs, labels, rate):
        '''
        one iteraion of training all the training data
        go through of all the training data and update the weights after scanning each sample
        :param input_vecs: 
        :param labels: 
        :param rate: 
        :return: 
        '''
        samples = zip(input_vecs, labels)

        for (input_vec, label) in samples:
            # print(np.shape((np.reshape(input_vec, (len(input_vec), 1)))))
            input_vec = np.reshape(input_vec, (1, len(input_vec)))
            output = self.calc_output(input_vec)
            # print(output)
            self.update_weights(input_vec, output, label, rate)

    def update_weights(self, input_vec, output, label, rate):
        '''
        update weights by gradient descent
        :param input_vec: 
        :param output: 
        :param label: 
        :param rate: 
        :return: 
        '''
        # print(output)
        error = label - output
        # print(type(error[0]))
        # self.weights = map(
        #     lambda x_w: x_w[1] + rate * error * x_w[0],
        #     zip(input_vec, self.weights)
        # )
        # print(np.shape(input_vec))
        # tmp = error[0] * input_vec.T
        # print(np.shape(input_vec))
        # print(np.shape(self.weights))
        self.weights = self.weights + error[0] * input_vec.T * rate


        self.bias = error * rate
        # print(error)
        print(self.weights.T)