import time

import numpy

import theano
import theano.tensor as T

from pylab import *

import scores

from logistic_regression_tanh import LogisticRegressionCrossEnt

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionCrossEnt(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.euclidean_loss = self.logRegressionLayer.euclidean_loss

        # same holds for the function computing the score of the model
        self.scores = self.logRegressionLayer.scores

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def train_mlp(train_labels, train_samples, hyperparams):

    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    n_epochs = hyperparams['n_epochs']
    lambda_reg = hyperparams['lambda_reg']
    num_hidden = hyperparams['num_hidden']

    borrow=True

    # Split data into training and validation here
    # TODO potential shuffle after split and not before
    arr = numpy.arange(train_labels.shape[0])
    numpy.random.shuffle(arr)

    cutoff = int(train_labels.shape[0] * 0.9)

    train_samples_x = train_samples[arr[0:cutoff],:]
    valid_samples_x = train_samples[arr[cutoff:],:]

    if len(train_labels.shape) == 1:
        train_samples_y = train_labels[arr[0:cutoff]]
        valid_samples_y = train_labels[arr[cutoff:]]

        train_samples_y.shape = (train_samples_y.shape[0], 1)
        valid_samples_y.shape = (valid_samples_y.shape[0], 1)
    else:
        train_samples_y = train_labels[arr[0:cutoff], :]
        valid_samples_y = train_labels[arr[cutoff:], :]

    train_set_x = theano.shared(numpy.asarray(train_samples_x, dtype=theano.config.floatX), borrow=borrow)
    train_set_y = theano.shared(numpy.asarray(train_samples_y, dtype=theano.config.floatX), borrow=borrow)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                           # [int] labels

    num_out = train_set_y.shape[1].eval()
    num_in = train_samples_x.shape[1]

    # construct the logistic regression class
    # classifier = LogisticRegressionCrossEnt(input=x, n_in=num_in, n_out=num_out, lambda_reg=lambda_reg)

    # for random weight intialisation
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=num_in,
                     n_hidden=num_hidden, n_out=num_out)
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y) + lambda_reg * classifier.L2_sqr

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    #print '... training the model'
    # early-stopping parameters
    patience = 100000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.9995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_params = None
    best_validation_score = -numpy.inf
    test_loss = 0.
    start_time = time.clock()

    validation_scores = numpy.array([])

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            #print 'Minibatch cost - %.4f' % minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                # Evaluate the model on all the minibatches
                _, _, _, _, f1, prec, rec = test_mlp(valid_samples_y, valid_samples_x, classifier.params)
                curr_f1 = numpy.mean(f1)

                W = classifier.params[0].eval()
                if numpy.isnan(numpy.sum(W)):
                    best_validation_score = 0
                    epoch = n_epochs
                    break

                validation_scores = numpy.hstack([validation_scores, curr_f1])

                print 'Epoch - %d - F1 - %f' % (epoch, curr_f1)

                # if we got the best validation score until now
                if curr_f1 > best_validation_score:

                    # Improve patience if loss improvement is good enough
                    if curr_f1 > best_validation_score / improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_score = curr_f1

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print 'Optimization complete with best validation score of %f ' % best_validation_score
    print 'The code run for %d epochs, with %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))

    #plot(validation_scores)
    #show()

    return classifier.params

def train_mlp_probe(train_labels, train_samples, test_labels, test_samples, hyperparams):

    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    n_epochs = hyperparams['n_epochs']
    lambda_reg = hyperparams['lambda_reg']
    num_hidden = hyperparams['num_hidden']

    decay = 0.01

    borrow=True

    # Split data into training and validation here
    # TODO potential shuffle after split and not before
    arr = numpy.arange(train_labels.shape[0])
    numpy.random.shuffle(arr)

    cutoff = int(train_labels.shape[0] * 0.9)

    train_samples_x = train_samples[arr[0:cutoff],:]
    valid_samples_x = train_samples[arr[cutoff:],:]

    if len(train_labels.shape) == 1:
        train_samples_y = train_labels[arr[0:cutoff]]
        valid_samples_y = train_labels[arr[cutoff:]]

        train_samples_y.shape = (train_samples_y.shape[0], 1)
        valid_samples_y.shape = (valid_samples_y.shape[0], 1)
    else:
        train_samples_y = train_labels[arr[0:cutoff], :]
        valid_samples_y = train_labels[arr[cutoff:], :]

    train_set_x = theano.shared(numpy.asarray(train_samples_x, dtype=theano.config.floatX), borrow=borrow)
    train_set_y = theano.shared(numpy.asarray(train_samples_y, dtype=theano.config.floatX), borrow=borrow)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                           # [int] labels

    num_out = train_set_y.shape[1].eval()
    num_in = train_samples_x.shape[1]

    # construct the logistic regression class
    # classifier = LogisticRegressionCrossEnt(input=x, n_in=num_in, n_out=num_out, lambda_reg=lambda_reg)

    # for random weight intialisation
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=num_in,
                     n_hidden=num_hidden, n_out=num_out)
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.euclidean_loss(y) + lambda_reg * classifier.L2_sqr

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    #print '... training the model'
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_params = None
    best_validation_score = -numpy.inf
    test_loss = 0.
    start_time = time.clock()

    validation_scores = numpy.array([])
    test_scores = numpy.array([])

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            #print 'Minibatch cost - %.4f' % minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                # Evaluate the model on all the minibatches

                _, _, _, _, f1, prec, rec = test_mlp(valid_samples_y, valid_samples_x, classifier.params)

                _, _, _, _, f1_t, prec_t, rec_t = test_mlp(test_labels, test_samples, classifier.params)
                #test_mlp(test_labels, test_samples, model):

                curr_f1 = numpy.mean(f1)
                curr_f1_test = numpy.mean(f1_t)

                print 'Epoch - %d, F1 - %f, F1(test) %f' % (epoch, curr_f1, curr_f1_test)

                if numpy.isnan(curr_f1):
                    best_validation_score = 0
                    epoch = n_epochs
                    break

                W = classifier.params[0].eval()
                if numpy.isnan(numpy.sum(W)):
                    best_validation_score = 0
                    epoch = n_epochs
                    break

                validation_scores = numpy.hstack([validation_scores, curr_f1])
                test_scores = numpy.hstack([test_scores, curr_f1_test])

                # if we got the best validation score until now
                if curr_f1 > best_validation_score:

                    # Improve patience if loss improvement is good enough
                    if curr_f1 > best_validation_score / improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_score = curr_f1

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print 'Optimization complete with best validation score of %f ' % best_validation_score
    print 'The code run for %d epochs, with %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))

    #plot(validation_scores)
    #plot(test_scores)
    show()

    return classifier.params


def test_mlp(test_labels, test_samples, model):

    W1 = model[0].eval()
    b1 = model[1].eval()

    W2 = model[2].eval()
    b2 = model[3].eval()

    tanh_out = numpy.tanh(numpy.dot(test_samples, W1) + b1)

    exp_in = numpy.dot(tanh_out, W2) + b2

    predictions = 1./(1+numpy.exp(- exp_in))

    predictions = predictions > 0.5

    test_l = test_labels.astype('int32')

    f1s, precisions, recalls = scores.FERA_class_score(predictions, test_l)

    return numpy.mean(f1s), numpy.mean(precisions), numpy.mean(recalls), predictions, f1s, precisions, recalls