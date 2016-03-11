import numpy as np


def learn(X, y, W):
    pass


def loss(X, y, W, reg):
    loss = np.sum((X.dot(W) - y) ** 2) / X.shape[0]
    grad = np.sum(
        (X.dot(W) - y).reshape(X.shape[0], 1) * X, axis=0) / X.shape[0] + reg * W
    return loss, grad


lr = 0.005
gamma = 0.1
step = 1000
display = 2000


def sgd(X, y, W, batch_size=10, iters=100000):
    global lr
    for it in range(iters):
        if it % step == 0:
            lr *= gamma
        chosen_ones = np.random.choice(X.shape[0], batch_size)
        X_batch = X[chosen_ones]
        y_batch = y[chosen_ones]
        # print X_batch, y_batch
        L, grad = loss(X_batch, y_batch, W, 0.0001)
        if it % display == 0:
            print L, W

        W -= lr * grad


# 2x1
W_God = np.array([3.2, 1.0, 4.135])
# Nx2
X = np.array(
    np.arange(0, 10, 0.05)
).reshape(100, 2)
# exit(0)
# X + bias
X = np.hstack([X, np.ones((X.shape[0], 1))])
# N
y_God = X.dot(W_God)

W = np.array([0.1, 0.3, 0.4])
# print loss(X, y_God, W)
# print loss(X, y_God, W_God)

sgd(X, y_God, W)
# print X
# print y_God
# print 2*3.2 + 3*4.135 + 3.1
