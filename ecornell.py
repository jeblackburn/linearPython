import numpy as np
# from sympy import *


def euclidean_distance_matrix():
    pass
    # X = Matrix([[1, 2],
    #             [3, 4]])
    # Z = Matrix([[1, 4],
    #             [2, 5],
    #             [3, 6]])
    # G = X * Z.T
    # print(G)
    # S = Matrix([[5, 5, 5],
    #             [25, 25, 25]])
    # print(S)
    # R = Matrix([[17, 29, 45],
    #             [17, 29, 45]])
    # print(R)
    # F = S + R - 2 * G
    # print(F)

    # print(np.vstack(F, axis=0))


def innerproduct(X, Z=None):
    '''
    function innerproduct(X,Z)

    Computes the inner-product matrix.
    Syntax:
    D=innerproduct(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d

    Output:
    Matrix G of size nxm
    G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]

    call with only one input:
    innerproduct(X)=innerproduct(X,X)
    '''
    if Z is None:  # case when there is only one input (X)
        Z = X

    return np.dot(X, Z.T)


def calculate_S(X, n, m):
    '''
    function calculate_S(X)

    Computes the S matrix.
    Syntax:
    S=calculate_S(X)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    n: number of rows in X
    m: output number of columns in S

    Output:
    Matrix S of size nxm
    S[i,j] is the inner-product between vectors X[i,:] and X[i,:]
    '''
    assert n == X.shape[0]

    return np.repeat(np.sum(np.square(X), axis=1), m).reshape(n, m)


def calculate_R(Z, n, m):
    '''
    function calculate_R(Z)

    Computes the R matrix.
    Syntax:
    R=calculate_R(Z)
    Input:
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    n: output number of rows in Z
    m: number of rows in Z

    Output:
    Matrix R of size nxm
    R[i,j] is the inner-product between vectors Z[j,:] and Z[j,:]
    '''
    assert m == Z.shape[0]

    return np.repeat(np.sum(np.square(Z), axis=1).reshape(1, m), n, axis=0)


def l2distance(X, Z=None):
    '''
    function D=l2distance(X,Z)

    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d

    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)

    call with only one input:
    l2distance(X)=l2distance(X,X)
    '''
    if Z is None:
        Z = X

    n, d1 = X.shape
    m, d2 = Z.shape
    assert (d1 == d2), "Dimensions of input vectors must match!"

    Dsquared = calculate_S(X, n, m) + calculate_R(Z, n, m) - 2 * innerproduct(X, Z)
    return np.sqrt(Dsquared.clip(min=0))


def findknn(xTr, xTe, k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);

    Finds the k nearest neighbors of xTe in xTr.

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """

    D = l2distance(xTr, xTe)

    k_nearest_neighbors = np.resize(np.argsort(D, axis=0), (k, D.shape[1]))

    k_distances = np.resize(np.sort(D, axis=0), (k, D.shape[1]))
    return k_nearest_neighbors, k_distances


def accuracy(truth, preds):
    c = truth == preds
    print(c)
    print(np.count_nonzero(c))
    print(np.size(c))
    return np.float64(np.count_nonzero(c)/np.size(c))


def knnclassifier(xTr, yTr, xTe, k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);

    k-nn classifier

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    yTr = n-dimensional vector of labels
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:

    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # fix array shapes
    yTr = yTr.flatten()

    k_indices, k_distances = findknn(xTr, xTe, k)
    print(k_indices)
    # Take each set of k nearest neighbors and look them up in yTr
    # Find the mode of the resulting array

    # xTr_nearest = k_indices.T[0, :]
    # print(yTr[xTr_nearest])

    from scipy.stats import mode as spmode

    def foo(a):
        print(a)
        nearest_labels = yTr[a]
        return spmode(nearest_labels)[0][0]

    selected_neighbors = np.apply_along_axis(foo, axis=0, arr=k_indices)
    print(selected_neighbors)

    # YOUR CODE HERE
    # raise NotImplementedError()


def mode(a):
    number_of_buckets = np.max(a) + 1
    histogram = np.histogram(a, number_of_buckets)[0]
    return histogram.argsort()[-1]


if __name__ == '__main__':
    # euclidean_distance_matrix()
    # n = 2
    # m = 3
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    # print(calculate_S(X, n, m))

    # n = 2
    # m = 3
    Z = np.array([[1, 4],
                  [2, 5],
                  [3, 6]])

    # D = l2distance(X, Z)
    # print(D)

    knnclassifier(X, np.array([7, 8, 9]), Z, 2)
    # print(calculate_R(Z, n, m))


    # print(np.argsort(D, axis=0))
    # print(np.sort(D, axis=0))

    # truth = np.array([1, 2, 3, 4])
    # preds = np.array([1, 2, 3, 0])
    # assert accuracy(truth, preds) == 0.75
    # a = [2, 1, 1, 3, 0, 0, 3, 0, 1, 0, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # print(mode(a))
