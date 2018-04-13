import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LogitRegression:

    def train_logit_regression(self, X_train, labels_train, iterations, ita):
        '''
        
        :param X_train: train data set
        :param labels_train: train labels
        :param iterations: number of iteration
        :param ita: learning rate
        :return: weights calculated by gradient descent
        '''

        # sampleNum, featureNum = np.shape(X_train)  # row number of train dataset is sample number, column number is features
        featureNum = len(X_train)
        weights = np.ones((featureNum, 1))  # set default weights as 1

        output = sigmoid(np.dot(X_train, weights))
        # print(type(output[0]))
        error = float(labels_train) - float(output[0])
        print(type(error))
        weights = weights + ita * X_train.T * error
        print(weights)

        return weights

    def test_accuarcy(self, X_test, labels_test, weights):
        numSamples, numFeatures = np.shape(X_test)
        matchCount = 0
        for i in range(numSamples):
            predict = sigmoid(X_test[i, :] * weights)[0, 0] > 0.5
            if predict == bool(labels_test[i, 0]):
                matchCount += 1
        accuracy = float(matchCount) / numSamples
        return accuracy

