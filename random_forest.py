import time

import matplotlib.pyplot as plt
import numpy as np
# from pylab import *
from numpy.matlib import repmat
from numpy.random import default_rng
from scipy.io import loadmat

from RegressionTree import RegressionTree


# from helper import *


def spiraldata(N=300):
    r = np.linspace(1, 2 * np.pi, N)
    xTr1 = np.array([np.sin(2. * r) * r, np.cos(2 * r) * r]).T
    xTr2 = np.array([np.sin(2. * r + np.pi) * r, np.cos(2 * r + np.pi) * r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1]) * 0.2

    xTe = xTr[::2, :]
    yTe = yTr[::2]
    xTr = xTr[1::2, :]
    yTr = yTr[1::2]

    return xTr, yTr, xTe, yTe


def iondata():
    data = loadmat("data/ionosphere.data")
    xTrIon = data['xTr'].T
    yTrIon = data['yTr'].flatten()
    xTeIon = data['xTe'].T
    yTeIon = data['yTe'].flatten()

    return xTrIon, yTrIon, xTeIon, yTeIon


iondata()


def forest(xTr, yTr, m, maxdepth=np.inf):
    """Creates a random forest.

    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree

    Output:
        trees: list of decision trees of length m
    """
    n, d = xTr.shape
    trees = []

    def create_tree():
        indices = np.random.randint(n, size=n)
        t = RegressionTree(depth=maxdepth)
        training_set = xTr[indices]
        training_labels = yTr[indices]
        t.fit(training_set, training_labels)
        return t

    #     v_create_tree = np.vectorize(create_tree)

    for x in range(m):
        trees.append(create_tree())

    return trees


def eval_forest(trees, X):
    """Evaluates X using trees.

    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector

    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n, d = X.shape

    results = np.zeros((m, n))
    #     print(m, n, results.shape)
    for t_idx in range(m):
        Y = trees[t_idx].predict(X)
        results[t_idx] = Y

    pred = np.mean(results, axis=0)
    return pred


def GBRT(xTr, yTr, m, maxdepth=4, alpha=0.1):
    """Creates GBRT.

    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        alpha:    learning rate for the GBRT


    Output:
        trees: list of decision trees of length m
        weights: weights of each tree
    """

    n, d = xTr.shape
    trees = []
    weights = []

    # Make a copy of the ground truth label
    # this will be the initial ground truth for our GBRT
    # This should be updated for each iteration
    t = np.copy(yTr)

    H = np.zeros(n)

    for i in range(m):
        t = yTr - H
        tree = RegressionTree(depth=maxdepth)
        tree.fit(xTr, t)
        score = tree.predict(xTr)
        H = H + np.multiply(score, alpha)

        trees.append(tree)
        weights.append(alpha)

    #     predictions = evalboostforest(trees, xTr, weights)

    return trees, weights


def evalboostforest(trees, X, alphas=None):
    """Evaluates X using trees.

    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector

    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n, d = X.shape

    if alphas is None:
        alphas = np.ones(m) / len(trees)

    results = np.zeros((m, n))
    for i in range(m):
        p_iter = trees[i].predict(X)
        results[i] = np.multiply(p_iter, alphas[i])

    pred = np.sum(results, axis=0)
    return pred


def visclassifier(fun, xTr, yTr, newfig=True):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """

    yTr = np.array(yTr).flatten()

    symbols = ["ko", "kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    # get the unique values from labels array
    classvals = np.unique(yTr)

    if newfig:
        plt.figure()

    # return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]), res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]), res)

    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # test all of these points on the grid
    testpreds = fun(xTe)

    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly

    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # creates x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c, 0],
                    xTr[yTr == c, 1],
                    marker=marker_symbols[idx],
                    color='k'
                    )

    plt.axis('tight')
    # shows figure and blocks
    plt.show()


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain, yTrain, w, b, M, Q, trees, weights

    if event.key == 'shift':
        Q += 10
    else:
        Q += 1
    Q = min(Q, M)

    classvals = np.unique(yTrain)

    # return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(0, 1, res)
    yrange = np.linspace(0, 1, res)

    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # get forest

    fun = lambda X: evalboostforest(trees[:Q], X, weights[:Q])
    # test all of these points on the grid
    testpreds = fun(xTe)
    trerr = np.mean(np.sign(fun(xTrain)) == np.sign(yTrain))

    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)

    plt.cla()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # fill in the contours for these predictions
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c, 0], xTrain[yTrain == c, 1], marker=marker_symbols[idx], color='k')
    plt.show()
    plt.title('# Trees: %i Training Accuracy: %2.2f' % (Q, trerr))


def demo_onclick_forest():
    xTrSpiral, yTrSpiral, xTeSpiral, yTeSpiral = spiraldata(150)
    xTrIon, yTrIon, xTeIon, yTeIon = iondata()

    xTrain = xTrSpiral.copy() / 14 + 0.5
    yTrain = yTrSpiral.copy()
    yTrain = yTrain.astype(int)

    # Hyper-parameters (feel free to play with them)
    M = 50
    alpha = 0.05
    depth = 5
    trees, weights = GBRT(xTrain, yTrain, M, alpha=alpha, maxdepth=depth)
    Q = 0

    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick_forest)
    print('Click to add a tree.')
    plt.title('Click to start boosting on the spiral data.')
    visclassifier(lambda X: np.sum(X, 1) * 0, xTrain, yTrain, newfig=False)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


if __name__ == '__main__':
    rng = default_rng(int(time.time() * 10000))
    a = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [7, 7, 7]
    ])
    # print(a[0])

    np.zeros()
    print(np.mean(a, axis=0))

    cols, rows = np.meshgrid(np.arange(100), np.arange(100))
    # print(cols)
    # print(rows)
    # c = rng.integers(100, size=100)
    # c = np.random.randint(100, size=100)
    # print(c)
    # print(cols[c, :])
    # print(a)
    # print(np.round(rng.random(1000)))
