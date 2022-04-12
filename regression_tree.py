import numpy as np

xor4 = np.array([[1, 1, 1, 1],
                 [1, 1, 1, 0],
                 [1, 1, 0, 1],
                 [1, 1, 0, 0],
                 [1, 0, 1, 1],
                 [1, 0, 1, 0],
                 [1, 0, 0, 1],
                 [1, 0, 0, 0],
                 [0, 1, 1, 1],
                 [0, 1, 1, 0],
                 [0, 1, 0, 1],
                 [0, 1, 0, 0],
                 [0, 0, 1, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0]])
yor4 = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])


def sqimpurity(yTr):
    """
    Computes the squared loss impurity (variance) of the labels.

    Input:
        yTr: n-dimensional vector of labels

    Output:
        squared loss impurity: weighted variance/squared loss impurity of the labels
    """
    N, = yTr.shape
    # assert N > 0  # must have at least one sample
    if N == 0:
        raise ValueError("No samples in training set", yTr.shape)
    meanY = np.mean(yTr)
    return sum((yTr - meanY) ** 2)


def sqsplit(xTr, yTr):
    """
    Finds the best feature, cut value, and impurity for a split of (xTr, yTr) based on squared loss impurity.

    Input:
        xTr: n x d matrix of data points
        yTr: n-dimensional vector of labels

    Output:
        feature:  index of the best cut's feature (keep in mind this is 0-indexed)
        cut:      cut-value of the best cut
        bestloss: squared loss impurity of the best cut
    """
    n, d = xTr.shape
    assert d > 0  # must have at least one dimension
    assert n > 1  # must have at least two samples
    best_loss = np.inf
    best_feature = np.inf
    best_cut = np.inf

    for feature in range(d):
        indices = np.argsort(xTr[:, feature])
        x = xTr[indices]
        y = yTr[indices]
        for i in range(n - 1):
            if x[i, feature] != x[i + 1, feature]:
                t = (x[i, feature] + x[i + 1, feature]) / 2
                # s_l = x[:i]
                # s_r = x[i + 1:]
                y_l = y[:i+1]
                y_r = y[i + 1:]
                impurity = sqimpurity(y_l) / y_l.shape[0] + sqimpurity(y_r) / y_r.shape[0]
                if impurity < best_loss:
                    best_cut = t
                    best_feature = feature
                    best_loss = impurity
    #                     print("Breaking point: feature:", feature, "left:", s_l, "right", s_r, "bestloss", best_loss)

    return best_feature, best_cut, best_loss
