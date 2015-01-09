import numpy
import time

import theano
import theano.tensor as T

from pylab import *

import scores


class LogisticRegressionCrossEnt(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, lambda_reg=0):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute a matrix of class-membership probabilities in symbolic form
        self.p_y_given_x = (T.tanh(T.dot(input, self.W) + self.b) + 1) / 2

        self.lambda_reg = lambda_reg

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return T.mean(T.neg(y) * T.log(self.p_y_given_x) - (1 + T.neg(y)) * T.log(1 - self.p_y_given_x)) + \
               self.lambda_reg * T.sum(self.W ** 2)

    def scores(self, y):
        positive_preds = self.p_y_given_x > 0.5
        negative_preds = self.p_y_given_x <= 0.5

        neg_y = y < 1

        tps = T.sum(positive_preds * y, axis=0)
        fps = T.sum(positive_preds * neg_y, axis=0)
        tns = T.sum(negative_preds * neg_y, axis=0)
        fns = T.sum(negative_preds * y, axis=0)

        precisions = tps / (tps + fps)
        recalls = tps / (tps + fns)

        f1s = 2 * precisions * recalls / (precisions + recalls)

        return f1s, precisions, recalls

    def error(self, y):
        return T.mean(T.neq(T.argmax(self.p_y_given_x, axis=1), T.argmax(y, axis=1)))


def train_log_reg(train_labels, train_samples, hyperparams):
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    n_epochs = hyperparams['n_epochs']
    lambda_reg = hyperparams['lambda_reg']

    borrow = True

    # Split data into training and validation here
    # TODO potential shuffle after split and not before
    arr = numpy.arange(train_labels.shape[0])
    numpy.random.shuffle(arr)

    cutoff = int(train_labels.shape[0] * 0.9)

    train_samples_x = train_samples[arr[0:cutoff], :]

    valid_samples_x = train_samples[arr[cutoff:], :]

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

    valid_set_x = theano.shared(numpy.asarray(valid_samples_x, dtype=theano.config.floatX), borrow=borrow)
    valid_set_y = theano.shared(numpy.asarray(valid_samples_y, dtype=theano.config.floatX), borrow=borrow)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
    # [int] labels

    num_out = train_set_y.shape[1].eval()
    num_in = train_samples_x.shape[1]

    # construct the logistic regression class
    classifier = LogisticRegressionCrossEnt(input=x, n_in=num_in, n_out=num_out, lambda_reg=lambda_reg)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    score_validate_model = theano.function(inputs=[index],
                                           outputs=classifier.scores(y),
                                           givens={
                                               x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                               y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

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
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.9  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    # go through this many minibatches before checking the network on the validation set

    best_params = None
    best_validation_score = -numpy.inf
    start_time = time.clock()

    validation_scores = numpy.array([])

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            # minibatch_avg_cost, minibatch_f1s = train_model(minibatch_index)
            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                # Evaluate the model on all the minibatches

                validation_scores_c = []
                for i in xrange(n_valid_batches):
                    f1s, precisions, recalls = score_validate_model(i)
                    validation_scores_c.append(f1s)

                curr_f1 = numpy.mean(validation_scores_c)

                if numpy.isnan(curr_f1):
                    curr_f1 = 0

                validation_scores = numpy.hstack([validation_scores, curr_f1])

                # if we got the best validation score until now
                if curr_f1 > best_validation_score:
                    # improve patience if loss improvement is good enough
                    if curr_f1 > best_validation_score / improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_score = curr_f1

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print 'Optimization complete with best validation score of %f ' % best_validation_score
    print 'The code run for %d epochs, with %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))

    return classifier.W.eval(), classifier.b.eval()


def test_log_reg(test_labels, test_samples, model):
    W = model[0]
    b = model[1]

    predictions = (numpy.tanh((numpy.dot(test_samples, W) + b)) + 1) / 2
    predictions = predictions > 0.5

    test_l = test_labels.astype('int32')
    test_l = test_l[:, 0]

    f1s, precisions, recalls = scores.FERA_class_score(predictions, test_l)

    return numpy.mean(f1s), numpy.mean(precisions), numpy.mean(recalls), predictions