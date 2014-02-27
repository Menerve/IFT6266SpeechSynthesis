__author__ = 'thomas'

import numpy as np


class MultilayerPerceptronRegression:
    def __init__(self, ninp, nhid, nout, l2, lr):
        # mlp hyper-parameters
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.l2 = l2
        self.lr = lr
        self.ha = np.zeros(self.nhid)
        self.hs = np.zeros(self.nhid)
        self.os = np.zeros(self.nout)

        # parameters initialisation
        # layer 1
        w1bound = 1 / np.sqrt(self.ninp)
        self.w1 = np.asarray(np.random.uniform(low=-w1bound, high=w1bound, size=(self.nhid, self.ninp)))
        self.b1 = np.zeros((self.nhid, 1))

        # layer 2
        w2bound = 1 / np.sqrt(self.nhid)
        self.w2 = np.asarray(np.random.uniform(low=-w2bound, high=w2bound, size=(self.nout, self.nhid)))
        self.b2 = np.zeros((self.nout, 1))

    def fprop(self, inputs):
        self.ha = np.dot(self.w1, inputs.transpose()) + self.b1
        self.hs = np.tanh(self.ha)
        oa = np.dot(self.w2, self.hs) + self.b2
        self.os = oa

        return self.os

    def bprop(self, inputs, targets):
        # Gradients
        grad_oa = 2 * (self.os - targets)
        grad_b2 = grad_oa
        grad_w2 = np.dot(grad_oa, self.hs.transpose())
        grad_hs = np.dot(self.w2.transpose(), grad_oa)
        grad_ha = grad_hs * (1. - (self.hs ** 2))
        grad_b1 = grad_ha
        grad_w1 = np.dot(grad_ha, inputs)

        return [grad_w1, grad_b1, grad_w2, grad_b2]

    def compute_loss(self, train_set, valid_set, test_set):

        train_error = np.sum((self.fprop(train_set[:, :-1]) - train_set[:, -1]) ** 2) / train_set.shape[0]
        valid_error = np.sum((self.fprop(valid_set[:, :-1]) - valid_set[:, -1]) ** 2) / valid_set.shape[0]
        test_error = np.sum((self.fprop(test_set[:, :-1]) - test_set[:, -1]) ** 2) / test_set.shape[0]

        return [train_error, valid_error, test_error]

    def stats(self, train_set, valid_set, test_set):
        loss = self.compute_loss(train_set, valid_set, test_set)

        sets = ["Train", "Valid", "Test"]
        for i, loss_rate in enumerate(loss):
            print sets[i], "Loss rate:", loss_rate

        return loss

    def update(self, gradients):
        # regularization
        if self.l2 > 0:
            gradients[0] += 2. * self.l2 * self.w1
            gradients[2] += 2. * self.l2 * self.w2

        self.w1 -= self.lr * gradients[0]
        self.b1 -= self.lr * np.sum(gradients[1], axis=1).reshape((self.b1.shape[0], 1))
        self.w2 -= self.lr * gradients[2]
        self.b2 -= self.lr * np.sum(gradients[3], axis=1).reshape((self.nout, 1))

    def train(self, train_data, n_epochs, batch_size, test_data, valid_data):

        loss = np.zeros((n_epochs, 3))

        for j in range((n_epochs * train_data.shape[0]) / batch_size):

            start = j * batch_size % train_data.shape[0]
            batch_data = train_data[start: start + batch_size, :]
            # forward propagation
            self.fprop(batch_data[:, :-1])

            # back propagation
            grad_params = self.bprop(batch_data[:, :-1], batch_data[:, -1])

            self.update(grad_params)

            if (j + 1) * batch_size % train_data.shape[0] == 0 and j != 0:
                n_epoch = j / (train_data.shape[0] / batch_size) + 1
                print "Epoch: ", n_epoch
                loss[n_epoch - 1] = self.stats(train_data, valid_data, test_data)

        return loss

    def compute_predictions(self, test_data):
        sorties = self.fprop(test_data)
        return sorties.transpose()

