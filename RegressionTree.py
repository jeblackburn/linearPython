import numpy as np


class RegressionTree(object):
    def __init__(self, depth=np.inf, weights=None):
        self.depth = depth
        self.weights = weights
        self.root = None

    def fit(self, XTr, yTr):
        self.root = self.cart(XTr, yTr, self.depth, self.weights)

    def predict(self, XTr):
        if self.root is None:
            raise NotImplementedError('The tree is not fitted yet!')
        return self.evaltree(self.root, XTr)

    # noinspection PyMethodMayBeStatic
    def sqsplit(self, xTr, yTr, weights=None):
        """Finds the best feature, cut value, and loss value.
        
        Input:
            xTr:     n x d matrix of data points
            yTr:     n-dimensional vector of labels
            weights: n-dimensional weight vector for data points
        
        Output:
            feature:  index of the best cut\'s feature
            cut:      cut-value of the best cut
            bestloss: loss of the best cut
        """
        N, D = xTr.shape
        assert D > 0  # must have at least one dimension
        assert N > 1  # must have at least two samples
        if weights is None:  # if no weights are passed on, assign uniform weights
            weights = np.ones(N)
        weights = weights / sum(weights)  # Weights need to sum to one (we just normalize them)
        bestloss = np.inf
        feature = np.inf
        cut = np.inf

        # Begin Solution 
        for d in range(D):
            ii = xTr[:, d].argsort()  # sort data along that dimensions
            xs = xTr[ii, d]  # sorted feature values
            ws = weights[ii]  # sorted weights
            ys = yTr[ii]  # sorted labels

            # Initialize constants
            sL = 0.0  # mean squared label on left side
            muL = 0.0  # mean label on left side
            wL = 0.0  # total weight on left side
            sR = ws.dot(ys ** 2)  # mean squared label on right
            muR = ws.dot(ys)  # mean label on right
            wR = sum(ws)  # weight on right

            idif = np.where(np.abs(np.diff(xs, axis=0)) > np.finfo(float).eps * 100)[0]
            pj = 0

            for j in idif:
                deltas = np.dot(ys[pj:j + 1] ** 2, ws[pj:j + 1])
                delta_mu = np.dot(ws[pj:j + 1], ys[pj:j + 1])
                delta_w = np.sum(ws[pj:j + 1])

                sL += deltas
                muL += delta_mu
                wL += delta_w

                sR -= deltas
                muR -= delta_mu
                wR -= delta_w

                L = sL - muL ** 2 / wL
                R = sR - muR ** 2 / wR
                loss = L + R

                if loss < bestloss:
                    feature = d
                    cut = (xs[j] + xs[j + 1]) / 2
                    bestloss = loss

                pj = j + 1

        assert feature != np.inf and cut != np.inf

        return feature, cut, bestloss

    def cart(self, xTr, yTr, depth=np.inf, weights=None):
        """Builds a CART tree.
        
        The maximum tree depth is defined by "depth" (depth=2 means one split).
        Each example can be weighted with "weights".

        Args:
            xTr:      n x d matrix of data
            yTr:      n-dimensional vector
            depth: maximum tree depth
            weights:  n-dimensional weight vector for data points

        Returns:
            tree: root of decision tree
        """
        n, d = xTr.shape
        if weights is None:
            w = np.ones(n) / float(n)
        else:
            w = weights

        # Begin Solution
        index = np.arange(n)
        prediction = yTr.dot(w) / float(np.sum(w))
        if depth == 0 or np.all(yTr == yTr[0]) or np.max(np.abs(np.diff(xTr, axis=0))) < (np.finfo(float).eps * 100):
            # Create leaf Node
            return TreeNode(None, None, None, None, None, prediction)
        else:
            feature, cut, h = self.sqsplit(xTr, yTr, w)
            left_idx = index[xTr[:, feature] <= cut]
            right_idx = index[xTr[:, feature] > cut]

            left_w = w[left_idx]
            right_w = w[right_idx]
            left = self.cart(xTr[left_idx, :], yTr[left_idx], depth=depth - 1, weights=left_w)
            right = self.cart(xTr[right_idx, :], yTr[right_idx], depth=depth - 1, weights=right_w)
            currNode = TreeNode(left, right, None, feature, cut, prediction)
            left.parent = currNode
            right.parent = currNode

            return currNode

    def evaltree(self, root, xTe):
        """Evaluates xTe using decision tree root.
        
        Input:
            root: TreeNode decision tree
            xTe:  n x d matrix of data points
        
        Output:
            pred: n-dimensional vector of predictions
        """
        return self.evaltreehelper(root, xTe)

    def evaltreehelper(self, root, xTe, idx=None):
        """Evaluates xTe using decision tree root.
        
        Input:
            root: TreeNode decision tree
            xTe:  n x d matrix of data points
        
        Output:
            pred: n-dimensional vector of predictions
        """
        assert root is not None
        n = xTe.shape[0]
        pred = np.zeros(n)

        # TODO:
        if idx is None:
            idx = np.ones(n) == 1

        if root.left is None and root.right is None:
            return np.ones(sum(idx)) * root.prediction
        assert root.left is not None and root.right is not None
        feature, cutoff = root.cutoff_id, root.cutoff_val

        idxL = idx & (xTe[:, feature] <= cutoff)
        if not root.left.left and not root.left.right:
            pred[idxL] = root.left.prediction
        else:
            pred[idxL] = self.evaltreehelper(root.left, xTe, idxL)

        idxR = idx & (xTe[:, feature] > cutoff)
        if not root.right.left and not root.right.right:
            pred[idxR] = root.right.prediction
        else:
            pred[idxR] = self.evaltreehelper(root.right, xTe, idxR)
        return pred[idx]


class TreeNode(object):
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction
