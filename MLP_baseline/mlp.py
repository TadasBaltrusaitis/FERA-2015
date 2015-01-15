import time

import numpy

import theano
import theano.tensor as T

from pylab import *

import scores

from logistic_regression import LogisticRegressionCrossEnt

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

def train_mlp_probe(train_labels, train_samples, test_labels, test_samples, hyperparams):

    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    n_epochs = hyperparams['n_epochs']
    lambda_reg = hyperparams['lambda_reg']
    num_hidden = hyperparams['num_hidden']

    early_stopping = None
    if 'early_stopping' in hyperparams:
        early_stopping = hyperparams['early_stopping']

    adaptive_rate = True
    if 'adaptive_rate' in hyperparams:
        adaptive_rate = hyperparams['adaptive_rate']

    decay = 0.5

    borrow=True

    arr = numpy.arange(train_labels.shape[0])
    numpy.random.shuffle(arr)

    train_samples_x = train_samples[arr,:]
    train_samples_y = train_labels

    if len(train_labels.shape) == 1:
        train_samples_y.shape = (train_samples_y.shape[0], 1)

    train_samples_y = train_samples_y[arr,:]

    train_set_x = theano.shared(numpy.asarray(train_samples_x, dtype=theano.config.floatX), borrow=borrow)
    train_set_y = theano.shared(numpy.asarray(train_samples_y, dtype=theano.config.floatX), borrow=borrow)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                           # [int] labels
    learning_rate_t = T.scalar('learning_rate')

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
        updates.append((param, param - learning_rate_t * gparam))

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index, learning_rate_t],
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
    start_time = time.clock()

    validation_scores = numpy.array([])
    costs = numpy.array([])

    moving_scores = numpy.array([])
    moving_costs = numpy.array([])

    done_looping = False
    epoch = 0

    best_validation_score = -numpy.inf
    best_cost = numpy.inf

    validation_improved_in = 0
    cost_improved_in = 0

    W1_best = classifier.params[0].eval()
    b1_best = classifier.params[1].eval()

    W2_best = classifier.params[2].eval()
    b2_best = classifier.params[3].eval()

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        costs_epoch = numpy.array([])
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index, learning_rate)

            costs_epoch = numpy.hstack([costs_epoch, minibatch_avg_cost])

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

        W1 = classifier.params[0].eval()
        b1 = classifier.params[1].eval()

        W2 = classifier.params[2].eval()
        b2 = classifier.params[3].eval()

        # Evaluate the model on current eopch
        _, _, _, _, f1, prec, rec = test_mlp(test_labels, test_samples, (W1, b1, W2, b2))

        curr_f1 = numpy.mean(f1)

        validation_scores = numpy.hstack([validation_scores, curr_f1])

        if numpy.isnan(numpy.sum(W1)):
            print 'sth wrong'
            best_validation_score = 0
            break

        if numpy.isnan(curr_f1):
            print 'sth wrong 1'
            best_validation_score = 0
            break


        epoch_cost = np.mean(costs_epoch)
        costs = numpy.hstack([costs, epoch_cost])

        if(epoch <= 10):
            print 'Epoch - %d, cost - %f, F1 - %f' % (epoch, epoch_cost, curr_f1)
            moving_costs = numpy.hstack([moving_costs, epoch_cost])
            moving_scores = numpy.hstack([moving_scores, curr_f1])

        else:

            moving_costs = numpy.hstack([moving_costs, np.mean(costs[-10:])])
            moving_scores = numpy.hstack([moving_scores, np.mean(validation_scores[-10:])])

            if moving_costs[-1] < best_cost:
                best_cost = moving_costs[-1]
                cost_improved_in = 0
            else:
                cost_improved_in += 1

            if moving_scores[-1] > best_validation_score:

                W1_best = classifier.params[0].eval()
                b1_best = classifier.params[1].eval()

                W2_best = classifier.params[2].eval()
                b2_best = classifier.params[3].eval()

                best_validation_score = moving_scores[-1]
                score_improved_in = 0
                validation_improved_in = 0
            else:
                score_improved_in += 1
                validation_improved_in += 1

            if score_improved_in > 20:
                print 'Rate reduced'
                # Also move to previous stage
                learning_rate /= 1.5
                score_improved_in = 0

            # If the score has not improved in some time terminate early
            if validation_improved_in > 100:
                print 'Early termination'
                break

            print 'Epoch - %d, cost - %f (%f, %d), F1 - %f (%f, %d)' % \
               (epoch, epoch_cost, moving_costs[-1], cost_improved_in, curr_f1, moving_scores[-1], score_improved_in)

        # if(epoch > 10)
        #    costs

    end_time = time.clock()
    print 'Optimization complete with best validation score of %f ' % np.max(validation_scores)
    print 'The code run for %d epochs, with %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))

    #plot(moving_costs)
    #plot(moving_scores)
    #show()

    return (W1_best, b1_best, W2_best, b2_best)


def test_mlp(test_labels, test_samples, model):

    W1 = model[0]
    b1 = model[1]

    W2 = model[2]
    b2 = model[3]

    tanh_out = numpy.tanh(numpy.dot(test_samples, W1) + b1)

    exp_in = numpy.dot(tanh_out, W2) + b2

    predictions = 1./(1+numpy.exp(- exp_in))

    predictions = predictions > 0.5

    test_l = test_labels.astype('int32')

    f1s, precisions, recalls = scores.FERA_class_score(predictions, test_l)

    return numpy.mean(f1s), numpy.mean(precisions), numpy.mean(recalls), predictions, f1s, precisions, recalls