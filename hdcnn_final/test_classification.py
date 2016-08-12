#! /usr/bin/python
# -*- coding: utf8 -*-

"""
# *******************************************************
#  * Copyright (C) 2016 Hao Dong <hao.dong11@imperial.ac.uk>
#  *
#  * The code is a part of HDL project
#  *
#  * The code can not be copied, distributed, modified and/or
#  * used for research and/or any commercial use without the
#  * express permission of Hao Dong
#  *
# *******************************************************/
"""

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    ## you may want to change the path
    data_dir = ''   #os.getcwd() + '/lasagne_tutorial/'
    # print('data_dir > %s' % data_dir)

    X_train = load_mnist_images(data_dir+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(data_dir+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(data_dir+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_dir+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    ## you may want to plot one example
    # print('X_train[0][0] >', X_train[0][0].shape, type(X_train[0][0]))
    #         #  [[..],[..]]      (28, 28)      numpy.ndarray
    #         # plt.imshow 只支持 (28, 28)格式，不支持 (1, 28, 28),所以用 [0][0]
    # fig = plt.figure()
    # #plotwindow = fig.add_subplot(111)
    # plt.imshow(X_train[0][0], cmap='gray')
    # plt.title('A training image')
    # plt.show()

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluation(y_test, y_predict, n_classes):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    c_mat = confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)])
    f1    = f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)])
    f1_macro = f1_score(y_test, y_predict, average='macro')
    acc   = accuracy_score(y_test, y_predict)
    print('confusion matrix: \n',c_mat)
    print('f1-score:',f1)
    print('f1-score(macro):',f1_macro)   # same output with > f1_score(y_true, y_pred, average='macro')
    print('accuracy-score:', acc)
    return c_mat, f1, acc, f1_macro

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def get_update( update, loss, params, learning_rate):
    if update == 'momentum':
        updates = lasagne.updates.nesterov_momentum(           # Stochastic Gradient Descent (SGD) updates with Nesterov momentum
                loss, params, learning_rate=learning_rate, momentum=0.9)
    elif update == 'sgd':
        updates = lasagne.updates.sgd(loss, params, learning_rate = learning_rate)
    elif update == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate = learning_rate)
    else:
        raise Exception("Unknow update method")
    return updates

def visualize_CNN(CNN, second=10, saveable=True, name='cnn1_', fig_idx=39362):
    n_feature = CNN.shape[0]
    n_color = CNN.shape[1]
    n_row = CNN.shape[2]
    n_col = CNN.shape[3]
    row = int(np.sqrt(n_feature))
    col = int(np.ceil(n_feature/row))
    plt.ion()   # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for ir in range(1, row+1):
        for ic in range(1, col+1):
            if count > n_feature:
                break
            a = fig.add_subplot(col, row, count)
            plt.imshow(
                    np.reshape(CNN[count-1,:,:,:], (n_row, n_col)),
                    cmap='gray', interpolation="nearest")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def build_cnn(input_var=None, shape=(None, 1, 28, 28)):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with ? kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
        # 有 Conv1DLayer 和 Conv2DLayer http://lasagne.readthedocs.org/en/latest/modules/layers/conv.html
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=100, filter_size=(5, 5),    # stride=(1,1), pad=0, untie_biases=False
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    print('count_params >',lasagne.layers.count_params(network))
    return network

def main_classification():
    ## load MNIST data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test =  load_dataset() ##
    print('X_train.shape ', X_train.shape)  # (50000, 1, 28, 28)
    print('y_train.shape ', y_train.shape)  # (50000,)
    print('X_val.shape ', X_val.shape)      # (10000, 1, 28, 28)
    print('y_val.shape ', y_val.shape)      # (10000,)
    print('X_test.shape ', X_test.shape)    # (10000, 1, 28, 28)
    print('y_test.shape ', y_test.shape)    # (10000,)

    inputs = T.tensor4('inputs')     #  4-dimensional ndarray
    targets = T.ivector('targets')   #  1-dimension int32 ndarray

    network = build_cnn(inputs, shape=(None,1,28,28))

    learning_rate = 0.0001
    batch_size = 500
    loss_type = 'ce'
    print_freq = 10
    update = 'adam'
    n_epochs = 100
    pi_neuron = 4

    print("\nCompiling loss expression for CNN")
    ## train prediction, deterministic=False (default), enable Dropout
    prediction = lasagne.layers.get_output(network, deterministic=False)

    ## classification usually adopt cross-entropy
    ## regression usually adopt mean-squared-error
    mse = ((prediction - targets) ** 2 ).sum(axis=1).mean()
    ce = lasagne.objectives.categorical_crossentropy(prediction, targets).mean()

    ## L1_a: averaged activation value
    ## make the activation value more sparse, simular behaviour with vanilla sparse
    ## use for rectify layer
    # L1_a = 0
    # ## L2 weight decay
    # L2_w = 0
    # ## L1 weight decay
    # L1_w = 0
    ## Bast-a-Move neuron penalty (col of weight)
    Pi_col = 0
    ## Bast-a-Move feature penalty (row of weight)
    # Pi_row = 0

    params = lasagne.layers.get_all_params(network, trainable=True)
    print("  trainable params: %s" % params)
    for param in params:
        print("  %s: %s" % (str(param), param.get_value().shape))
    print("  batch_size: %d" % batch_size)

    # Pi_col = pi_neuron * T.sqrt(T.sum(params[0] ** 2, axis=0)).mean()   ## for W
    print("  pi_neuron: %f" % pi_neuron)
    if pi_neuron:
        Pi_col = pi_neuron * T.sqrt(
                    (params[0] ** 2).sum(axis=2).sum(axis=2)    # (32, 1, 5, 5) -> (32, 1, 5) -> (32, 1)
                    ).mean()


    if loss_type == "mse":
        print("  loss_type: mean-squared-error")
        loss = mse
    elif loss_type == "ce":
        print("  loss_type: cross-entropy")
        loss = ce
    else:
        raise Exception("Unknow loss_type")
    loss += Pi_col

    print("  update method: %s" % update)
    print("  learning_rate: %s" % learning_rate)
    updates = get_update(update, loss, params, learning_rate)

    ## deterministic=True, disable dropout
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    ## mean-squared-error for regression; cross-entropy for classification
    if loss_type == "mse":
        test_loss = ((test_prediction - targets) ** 2 ).sum(axis=1).mean()
    elif loss_type == "ce":
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets).mean()

    ## no accuracy for regression
    if loss_type == "ce":
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets), dtype=theano.config.floatX)

    train_fn = theano.function([inputs, targets], loss, updates=updates)

    if loss_type == "ce":
        val_fn = theano.function([inputs, targets], [test_loss, test_acc])
    else:
        val_fn = theano.function([inputs, targets], [test_loss])

    ## for evaluation on test data
    predict_fn = theano.function(inputs=[inputs], outputs=test_prediction)


    print("Start training")
    for epoch in range(n_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        acc = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            if loss_type == "ce":
                err, acc = val_fn(inputs, targets)
            else:
                err = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            ## you may want to check some penalty during training
            # print("  ", *check_fn(X_train[::20]))
            ## you may want to save the feature images

            cnn_1 = params[0].get_value()   # (32, 1, 5, 5)
            visualize_CNN(cnn_1, second=10, saveable=True, name='cnn1_'+str(epoch+1), fig_idx=34223)

    ## evaluate the network by using test data
    print("\nEvaluation")
    # predict_fn = mlpc.get_predict_function()
    ## if classification, you will got a one-hot-format
    y_predict = predict_fn(X_test)
    y_predict= np.argmax(y_predict, axis=1)
    evaluation(y_test, y_predict, 10)

    print('\nSave model as < model_hdcnn.mat > ...')
    import scipy
    from scipy.io import savemat, loadmat
    # params = mlpc.get_all_params()
    params_dict = {}
    for i in range(len(params)):
        params_dict[str(params[i])+str(i//2+1)] = params[i].get_value()
    savemat('model_hdcnn.mat', params_dict)


if __name__ == '__main__':
    main_classification()
