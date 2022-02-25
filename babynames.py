import math

import numpy as np


def naivebayesPY(X, Y):
    """
    naivebayesPY(X, Y)
    returns [pos,neg] (the Class Probabilities, ie the fraction of the training set that is pos/neg)

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """

    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1, 1]])
    n = len(Y)
    _, counts = np.unique(Y, return_counts=True)
    # print(values, counts)
    return counts[1] / n, counts[0] / n


def loglikelihood(boyposprob, girlposprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test

    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)

    Output:
        loglikelihood of each point in X_test (n)
    """

    row_numbers = np.arange(X_test.shape[0])
    X = np.insert(X_test, 0, row_numbers, axis=1)
    X = np.insert(X, 0, Y_test, axis=1)

    boy_rows = X[X[:, 0] == 1, 1:]
    girl_rows = X[X[:, 0] == -1, 1:]

    boy_row_numbers = boy_rows[:, 0:1].flatten()
    boy_rows = np.delete(boy_rows, 0, 1)

    girl_row_numbers = girl_rows[:, 0:1].flatten()
    girl_rows = np.delete(girl_rows, 0, 1)

    boynegprob = 1 - boyposprob
    boy_probs = boy_rows @ np.log(boyposprob) + (1 - boy_rows) @ np.log(boynegprob)
    boys_with_indexes = np.column_stack((boy_row_numbers, boy_probs))

    girlnegprob = 1 - girlposprob
    girl_probs = girl_rows @ np.log(girlposprob) + (1 - girl_rows) @ np.log(girlnegprob)
    girls_with_indexes = np.column_stack((girl_row_numbers, girl_probs))

    con = np.concatenate((boys_with_indexes, girls_with_indexes), axis=0)
    sorted_rows = con[con[:, 0].argsort()]
    combined = np.delete(sorted_rows, 0, 1)
    return combined.flatten()


def naivebayesPXY(X, Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]

    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)

    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """

    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2, d)), np.zeros((2, d))])
    Y = np.concatenate([Y, [-1, 1, -1, 1]])

    #     print(X)
    #     print(Y)
    X = np.insert(X, 0, Y, axis=1)
    #     print("Appended\n", X)
    boyRows = X[X[:, 0] == 1, :]
    girlRows = X[X[:, 0] == -1, :]
    #     print(boyRows)
    #     print(girlRows)

    boyProbabilities = np.mean(boyRows, axis=0)
    girlProbabilities = np.mean(girlRows, axis=0)

    return boyProbabilities[1:], girlProbabilities[1:]


def naivebayes_pred(pos, neg, posprob, negprob, X):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test

    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)

    Output:
        prediction of each point in X_test (n)
    """
    logpos = np.log(pos)
    logneg = np.log(neg)
    rowcount = X.shape[0]
    results = np.zeros(rowcount)

    # For each row in X, calculate the likelihood that it's a boy (np.ones)
    likelihood_boys = loglikelihood(posprob, negprob, X, np.ones(rowcount))
    # For each row in X, calculate the likelihood that it's a girl (np.ones*-1)
    likelihood_girls = loglikelihood(posprob, negprob, X, np.ones(rowcount) * -1)

    loglikelihood_ratio = (likelihood_boys + logpos) - (likelihood_girls + logneg)

    def convert_to_label(x):
        return 1 if x > 0 else -1

    return np.vectorize(convert_to_label)(loglikelihood_ratio)


def loglikelihood_grader(posprob, negprob, x_test, y_test):
    # calculate the likelihood of each of the point in x_test log P(x | y)
    n, d = x_test.shape
    loglikelihood = np.zeros(n)
    pos_ind = (y_test == 1)
    loglikelihood[pos_ind] = x_test[pos_ind]@np.log(posprob) + (1 - x_test[pos_ind])@np.log(1 - posprob)
    neg_ind = (y_test == -1)
    loglikelihood[neg_ind] = x_test[neg_ind]@np.log(negprob) + (1 - x_test[neg_ind])@np.log(1 - negprob)
    return loglikelihood


if __name__ == '__main__':
    X = np.array([[0, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1],
                  [1, 0, 0, 1, 1, 1],
                  [1, 1, 0, 1, 1, 0],
                  [0, 1, 0, 1, 0, 1]])
    Y = np.array([1, -1, -1, 1, 1])

    posY, negY = naivebayesPY(X, Y) # class probabilities
    posprobXY, negprobXY = naivebayesPXY(X, Y)
    preds = naivebayes_pred(posY, negY, posprobXY, negprobXY, X)
    print(preds)

    # Check that probabilities sum to 1
    # def naivebayesPY_test1():
    #     pos, neg = naivebayesPY(X, Y)
    #     return np.linalg.norm(pos + neg - 1) < 1e-5

    # Test the Naive Bayes PY function on a simple example
    def naivebayesPY_test2():
        x = np.array([[0, 1], [1, 0]])
        y = np.array([-1, 1])
        pos, neg = naivebayesPY(x, y)
        pos0, neg0 = .5, .5
        test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
        return test < 1e-5


    # Test the Naive Bayes PY function on another example
    def naivebayesPY_test3():
        x = np.array([[0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 0],
                      [1, 1, 1, 1, 0],
                      [0, 1, 1, 0, 1],
                      [1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 1, 0, 1]])
        y = np.array([1, -1, 1, 1, -1, -1, 1])
        pos, neg = naivebayesPY(x, y)
        pos0, neg0 = 5 / 9., 4 / 9.
        test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
        return test < 1e-5


    # Tests plus-one smoothing
    def naivebayesPY_test4():
        x = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
        y = np.array([1, 1])
        pos, neg = naivebayesPY(x, y)
        pos0, neg0 = 3 / 4., 1 / 4.
        test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
        return test < 1e-5

    # print(naivebayesPY_test2())

    # x = np.random.random((6, 4))
    # print(x)
    # print(x[:] > 0.5)
