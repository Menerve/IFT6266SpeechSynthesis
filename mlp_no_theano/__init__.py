__author__ = 'thomas'

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wv

from models.mlp_regression import MultilayerPerceptronRegression


if __name__ == '__main__':
    data = np.load('extr_FCJF0.npy')

    data_mean = np.mean(data)
    data_max = np.max(data)

    dataset = (data - data_mean) / data_max

    inds = range(dataset.shape[0])
    np.random.shuffle(inds)

    train_set = dataset[inds[:306200], :]
    test_set = dataset[inds[306200:344449], :]
    valid_set = dataset[inds[344449:], :]

    mu = 0.001
    n_hidden = [300]
    weight_decay = [0]
    n_epochs = 50
    batch_size = 200

    loss_rate = []

    for w in weight_decay:
        for n_dh in n_hidden:
            assert train_set.shape[0] % batch_size == 0

            print "Weight:", w, "Hidden number:", n_dh

            # Matrix implementation
            modelMlp = MultilayerPerceptronRegression(train_set.shape[1] - 1, n_dh, 1, w, mu)
            loss_rate = modelMlp.train(train_set, n_epochs, batch_size, test_set, valid_set)

            epochs = np.arange(n_epochs)
            train_loss_plot, = plt.plot(epochs, loss_rate[:, 0])
            valid_loss_plot, = plt.plot(epochs, loss_rate[:, 1])
            test_loss_plot, = plt.plot(epochs, loss_rate[:, 2])

            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.title('Error for train, valid and test sets')
            plt.legend([train_loss_plot, valid_loss_plot, test_loss_plot], ['Train set', 'Valid set', 'Test set'])
            plt.show()

            # Generate audio data
            predicted_data = np.zeros((50000, 1))
            predicted_data[0: 240] = np.reshape(dataset[1000, :-1], (240, 1))
            for i in range(240, 50000):
                predicted_data[i] = modelMlp.compute_predictions(predicted_data[i - 240:i].T)

            # Save in wav format
            output = np.int16(predicted_data * data_max + data_mean)
            wv.write('predicted_data.wav', 16000, output)

            # Plot the waveform
            plt.figure()
            plt.title('Predicted waveform')
            plt.plot(output)
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.show()
