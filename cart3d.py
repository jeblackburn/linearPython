import numpy as np

from regression_tree import sqsplit, xor4, yor4


class TreeNode(object):
    """
    Tree class.

    (You don't _need_ to add any methods or fields here but feel
    free to if you like. The tests will only reference the fields
    defined in the constructor below, so be sure to set these
    correctly.)
    """

    def __init__(self, left, right, feature, cut, prediction):
        # Check that all or no arguments are None
        node_or_leaf_args = [left, right, feature, cut]
        assert all([arg == None for arg in node_or_leaf_args]) or all([arg != None for arg in node_or_leaf_args])

        # Check that all None <==> leaf <==> prediction not None
        # Check that all non-None <==> non-leaf <==> prediction is None
        if all([arg == None for arg in node_or_leaf_args]):
            assert prediction is not None
        if all([arg != None for arg in node_or_leaf_args]):
            assert prediction is None

        self.left = left
        self.right = right
        self.feature = feature
        self.cut = cut
        self.prediction = prediction


def Leaf(prediction):
    return TreeNode(None, None, None, None, prediction)


def Branch(left, right, feature, cut):
    return TreeNode(left, right, feature, cut, None)


def cart(xTr, yTr):
    """
    Builds a CART tree.

    Input:
        xTr:      n x d matrix of data
        yTr:      n-dimensional vector

    Output:
        tree: root of decision tree
    """
    mean_y = np.mean(yTr)

    if np.all(np.isclose(yTr, mean_y)):
        # print("Creating leaf node - all yTr values are close to mean")
        return Leaf(mean_y)
    if np.all(np.isclose(xTr, xTr[0])):
        # print("Creating leaf node - all nodes have the same dimensions")
        return Leaf(mean_y)
    feature, cut, bestloss = sqsplit(xTr, yTr)
    # print("Solution", feature, cut, bestloss)
    left_indices = np.where(xTr[:, feature] <= cut)
    # print("Left", left_indices)
    right_indices = np.where(xTr[:, feature] > cut)
    # print("Right", right_indices)
    left_x = xTr[left_indices]
    right_x = xTr[right_indices]
    left_y = yTr[left_indices]
    right_y = yTr[right_indices]
    #     print("left", left_x)
    #     print("right", right_x)
    #     print("leftY", left_y)
    #     print("rightY", right_y)
    return Branch(cart(left_x, left_y), cart(right_x, right_y), feature, cut)


def evaltree(tree, xTe):
    """
    Evaluates testing points in xTe using decision tree root.

    Input:
        tree: TreeNode decision tree
        xTe:  m x d matrix of data points

    Output:
        pred: m-dimensional vector of predictions
    """
    m, d = xTe.shape
    preds = np.zeros(m)

    print("Shape", m, d)

    if tree.prediction is not None:
        print("Leaf Node with prediction", tree.prediction, "applying to nodes:", m)
        return np.repeat(tree.prediction, m)

    # left, right, feature, cut
    left_indices = np.where(xTe[:, tree.feature] <= tree.cut)
    print("Left indices", left_indices)
    right_indices = np.where(xTe[:, tree.feature] > tree.cut)
    print("Right indices", right_indices)

    left_predictions = evaltree(tree.left, xTe[left_indices])
    print("Left predictions", left_predictions)
    preds[left_indices] = left_predictions

    right_predictions = evaltree(tree.right, xTe[right_indices])
    print("Right", right_predictions)
    preds[right_indices] = right_predictions

    # return np.concatenate((right_predictions, left_predictions))
    return preds

t = cart(xor4, yor4)
xor4te = xor4 + (np.sign(xor4 - .5) * .1)

indices = np.arange(16)
np.random.shuffle(indices)
print(indices)
expected = yor4[indices]
inputDataSet = xor4te[indices, :]
# Check that shuffling and expanding the data doesn't affect the predictions
result = evaltree(t, inputDataSet)
assert np.all(np.isclose(result.astype(int), expected))



# a = TreeNode(None, None, None, None, 1)
# b = TreeNode(None, None, None, None, -1)
# c = TreeNode(None, None, None, None, 0)
# d = TreeNode(None, None, None, None, -1)
# e = TreeNode(None, None, None, None, -1)
# x = TreeNode(a, b, 0, 10, None)
# y = TreeNode(x, c, 0, 20, None)
# z = TreeNode(d, e, 0, 40, None)
# t = TreeNode(y, z, 0, 30, None)
# # Check that the custom tree evaluates correctly
# xTr = np.array([[45, 35, 25, 15, 5]]).T
# print("Input", xTr)
# result = evaltree(t, xTr)
# print("Result", result)
# print(np.all(np.isclose(
#     result,
#     np.array([-1, -1, 0, -1, 1]))))

# np.where(xor4[:,0] <= 0.4)
# xor4[:, 0]
# xor4[:, 0]
# cart(xor4, yor4)
# print(t)

# t = cart(xor4, yor4)
# print(t)
# print(np.concatenate((np.array([1, 2, 3]), np.array([4, 5, 6]))))
# yTe = DFSpreds(t)[:];
# Check that every label appears exactly once in the tree
# y.sort()
# # yTe.sort()
# print("Result", y)
# print("yTe", yTe)
# np.all(np.isclose(y, yTe))
