import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        margin = scores - correct_class_score + 1
        margin_mask = margin > 0
        margin_mask[y[i]] = False
        loss += np.sum(margin[margin_mask])
        correct_class_gradient = X[i] * np.sum(margin_mask)
        dW_one = np.zeros(W.shape)
        tmp = np.zeros(margin_mask.shape)
        tmp[margin_mask] = 1
        tmp = tmp.reshape(1, num_classes)
        dW_one = X[i].reshape(X.shape[1], 1).dot(tmp)
        dW_one[:, y[i]] = -correct_class_gradient
        dW += dW_one
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    ##########################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    ##########################################################################
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    ##########################################################################
    scores = X.dot(W)
    # 500x1
    correct_class_score = scores[
        range(scores.shape[0]), y.reshape(1, y.shape[0])].T
    # 500x10
    margins = scores - correct_class_score + 1
    margins[range(margins.shape[0]), y.reshape(1, y.shape[0])] = 0
    margins_mask = margins > 0
    margins = np.maximum(0, margins)
    # set correct class margin to 0
    loss = np.sum(margins) / X.shape[0]
    correct_class_gradient = X * \
        np.sum(margins > 0, axis=1).reshape((X.shape[0], 1))
    tmp = np.zeros(margins.shape)
    tmp[margins_mask] = 1
    dW = X.T.dot(tmp)
    for idx, ly in enumerate(y):
        dW[:, ly] -= correct_class_gradient[idx, :]
    dW = dW / X.shape[0]
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    ##########################################################################

    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    return loss, dW
