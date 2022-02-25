import numpy as np


def perceptron_update(x, y, w):
    return w.transpose() + y * x


def append_bias(x):
    rowcount = x.shape[0]
    # print("Rows", x)
    biases = np.atleast_2d(np.ones(rowcount)).T
    # print("ones", biases)
    return np.hstack((x, biases))


def linear_classifier(xs, w, b=None):
    """
    function preds=classify_linear(xs,w,b)

    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)

    Output:
    preds: predictions (1xn)
    """
    w = w.flatten()

    # This is my answer key.
    predictions_ans = [-1 if (w.transpose().dot(x) + (b or 0) < 0) else 1 for x in xs]
    dotproducts = w.transpose().dot(xs.T).T + (b or 0)
    convert_to_label = np.vectorize(lambda z: -1 if z + b < 0 else 1)
    predictions = convert_to_label(dotproducts)
    np.testing.assert_array_equal(predictions, predictions_ans)
    return predictions


if __name__ == "__main__":
    # messing around with atleast_2d
    # ones = np.array(np.ones(100))
    # print(ones)
    # print(ones.T)
    # print(np.atleast_2d(ones).T)

    xs = np.array([[1, 3], [-1, 4]])
    ys = np.array([1, -1])

    # print(xs, ys)
    with_biases = append_bias(xs)
    # print('Biases appended:', with_biases)

    w = np.zeros(with_biases.shape[1])  # with zero bias appended
    # print("Initial w: ", w)

    # convert to tuples
    points = list(zip(with_biases, ys))
    # print(points)

    idx = 0
    hyperplane_found = False
    while not hyperplane_found:
        # print("iteration", idx + 1)
        point_to_add = points[idx % len(points)]
        # print("w:", w)
        broken_point_pair = None
        for point_pair in points:
            distance = point_pair[1] * (w.transpose().dot(point_pair[0]))
            # print(f"p correct side for {point_pair}?", distance > 0, distance)
            if distance <= 0:
                broken_point_pair = point_pair
                break
        if not broken_point_pair:
            # print("Found it:", w)
            hyperplane_found = True
        if not hyperplane_found:
            # print("Adding to w: ", point_to_add)
            w = perceptron_update(point_to_add[0], point_to_add[1], w)
            idx += 1

    real_w = w[:-1]
    b = w[-1]
    print("w and b:", real_w, b)

    print("*******************************************************************")

    xs = np.array([
        [2, 2],
        [2, -2],
        [-2, -2],
        [-2, 2],
    ])
    results = linear_classifier(xs, real_w, b)
    print(results)
