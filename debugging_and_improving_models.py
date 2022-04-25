import numpy as np

from RegressionTree import RegressionTree


def toydata(OFFSET, N):
    """
    function [x,y]=toydata(OFFSET,N)
    
    This function constructs a binary data set. 
    Each class is distributed by a standard Gaussian distribution.
    INPUT: 
    OFFSET:  Class 1 has mean 0,  Class 2 has mean 0+OFFSET (in each dimension). 
    N: The function returns N data points ceil(N/2) are of class 2, the rest
    of class 1
    """
    NHALF = int(np.ceil(N/2))
    x = np.random.randn(N, 2)
    x[NHALF:, :] += OFFSET

    y = np.ones(N)
    y[NHALF:] *= 2

    jj = np.random.permutation(N)
    return x[jj, :], y[jj]


def normpdf(x, mu, sigma):
    """

    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    return np.exp(-0.5 * np.power((x - mu) / sigma, 2)) / (np.sqrt(2 * np.pi) * sigma)


def computeybar(xTe, OFFSET):
    """
    function [ybar]=computeybar(xTe, OFFSET);

    computes the expected label 'ybar' for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    ybar : a nx1 vector of the expected labels for vectors xTe
    noise:
    """
    sigma = 1
    class_1_mu = 0
    class_2_mu = OFFSET


    class1_px = normpdf(xTe, class_1_mu, sigma)
    class2_px = normpdf(xTe, class_2_mu, sigma)
    #     print(class1_px_given_y.shape, class1_px_given_y)
    #     print(class2_px_given_y.shape, class2_px_given_y)

    class1_aggregate_px = np.multiply(class1_px[:, 0], class1_px[:, 1])
    class2_aggregate_px = np.multiply(class2_px[:, 0], class2_px[:, 1])
    #     print(class2_aggregate_px.shape)
    ybar = (class1_aggregate_px + 2 * class2_aggregate_px) / (class1_aggregate_px + class2_aggregate_px)

    return ybar


def computenoise(xTe, yTe, OFFSET):
    """
    function noise=computenoise(xTe, OFFSET);

    computes the noise, or square mean of ybar - y, for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    noise:    : a scalar representing the noise component of the error of xTe
    """
    ybar = computeybar(xTe, OFFSET)
    deltas_squared = np.square(ybar - yTe)
    noise = np.sum(deltas_squared) / xTe.shape[0]
    return noise


def computehbar(xTe, depth, Nsmall, NMODELS, OFFSET):
    """
    function [hbar]=computehbar(xTe, sigma, lmbda, NSmall, NMODELS, OFFSET);

    computes the expected prediction of the average regression tree (hbar)
    for data set xTe.

    The regression tree should be trained using data of size Nsmall and is drawn from toydata with OFFSET


    The "infinite" number of models is estimated as an average over NMODELS.

    INPUT:
    xTe       | nx2 matrix, of n column-wise input vectors (each 2-dimensional)
    depth     | Depth of the tree
    NSmall    | Number of points to subsample
    NMODELS   | Number of Models to average over
    OFFSET    | The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.
    OUTPUT:
    hbar | nx1 vector with the predictions of hbar for each test input
    """
    n = xTe.shape[0]

    trainingset = np.zeros((n, NMODELS))
    for nm in range(NMODELS):
        xTr, yTr = toydata(OFFSET, Nsmall)
        tree = RegressionTree(depth=depth)
        tree.fit(xTr, yTr)
        trainingset[:, nm] = tree.predict(xTe)

    # YOUR CODE HERE
    hbar = np.mean(trainingset, axis=1)
    return hbar


def computevariance(xTe, depth, hbar, Nsmall, NMODELS, OFFSET):
    """
    function variance=computevbar(xTe,sigma,lmbda,hbar,Nsmall,NMODELS,OFFSET)

    computes the variance of classifiers trained on data sets from
    toydata.m with pre-specified "OFFSET" and
    with kernel regression with sigma and lmbda
    evaluated on xTe.
    the prediction of the average classifier is assumed to be stored in "hbar".

    The "infinite" number of models is estimated as an average over NMODELS.

    INPUT:
    xTe       : nx2 matrix, of n column-wise input vectors (each 2-dimensional)
    depth     : Depth of the tree
    hbar      : nx1 vector of the predictions of hbar on the inputs xTe
    Nsmall    : Number of samples drawn from toyData for one model
    NModel    : Number of Models to average over
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    vbar      : nx1 vector of the difference between each model prediction and the
                average model prediction for each input

    """
    n = xTe.shape[0]

    training_set = np.zeros((n, NMODELS))
    for nm in range(NMODELS):
        xTr, yTr = toydata(OFFSET, Nsmall)
        tree = RegressionTree(depth=depth)
        tree.fit(xTr, yTr)
        predictions = tree.predict(xTe)
        training_set[:, nm] = np.square(hbar - predictions)

    vbar = np.mean(training_set, axis=1)
    variance = np.mean(vbar)

    return variance


if __name__ == '__main__':
    x = np.array([
        [1, 1, 1],
        [2, 2, 2]
    ])

    print(np.mean(x, axis=1))

