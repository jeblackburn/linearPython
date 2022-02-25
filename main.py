import mpmath
import numpy
from sympy import *
from scipy.stats import mode


def rotate_3d():
    theta = mpmath.radians(225)
    transform2d = Matrix([
        # [1, -2, 5],
        # [1, 4, -2],
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)],
    ])
    # pprint(transform2d * Matrix([-1, 4]))
    transform3d_xaxis = Matrix([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta), cos(theta)],
    ])
    # pprint(transform3d_xaxis * Matrix([2, 0, -3]))
    transform3d_zazis = Matrix([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1],
    ])
    # pprint(transform3d_zazis * Matrix([-2, 3, -1]))


def rref(m):
    # m = Matrix([
    #     [-2, 1, 0, 4],
    #     [1, -3, 0, 1],
    #     [0, 4, -1, 2],
    #     [2, 0, 1, -3],
    # ])
    pprint(m.rref())
    # pprint(m.det())
    # pprint(m.adjugate())
    # pprint(m.inv())


def composition_txfms():
    s = Matrix([
        [2, -1, 1],
        [0, 0, 4],
        [-1, 1, 0],
    ])
    t = Matrix([
        [-3, 0, 0],
        [0, 2, 1],
        [0, 0, 4],
    ])
    st = t * s
    pprint(st * Matrix([2, -4, 0]))


def inverse():
    m = Matrix([
        [1, 1, -2],
        [2, 3, 1],
        [3, -3, 4]
    ])
    pprint(m)
    pprint(m.inv())
    # pprint(m.inv() * Matrix([-20, -66]))


def determinant(m):
    # m = Matrix([[1, 5, 0, -1], [3, -2, -1, 2], [-1, 1, 0, 3], [1, 3, 2, -2]])
    pprint(m.det())


def inverse(m):
    return m.inv()


def transpose(m):
    return m.transpose()


def orthogonal_complement(m: Matrix):
    mi = m.transpose()
    rref = mi.rref()
    pprint(rref)


if __name__ == '__main__':
    init_printing(use_unicode=True)

    a = numpy.array([1, 1, 1, 1, 3, 45, 6, 7, 6, 6, 76, ])
    pprint(mode(a)[0][0])

    # m = Matrix([
    #     [1, -2, 3, -1, 2],
    #     [-3, 6, -9, 3, -6],
    #     [-5, 9, -7, 4, 0],
    # ])
    # print('Q1:', m.rref())
    #
    m = Matrix([
        [2, -3, 6, -5, -6],
        [4, -5, 12, -11, -14],
        [2, -2, 6, -6, -8],
    ])
    n = Matrix([
        [4, -6, 1],
        [0, -8, 5],
        [1, 1, -2],
    ])

    # u = Matrix([1, 4, 8])
    w = Matrix([1, 3, 3, 4, 1])
    p1 = Matrix([1, 1, 1, 0, -1])
    p2 = Matrix([2, 1, 1, 1, 0])
    p3 = Matrix([1, -4, 2, -2, -2])
    p4 = Matrix([-1, -1, -2, 2, -4])
    p5 = Matrix([-1, 1, -2, 3, 1])

    # pprint(w.norm())
    # pprint(w.dot(p1))
    # print('xxxxxx')
    # pprint((w.dot(p1) - 12) / w.norm())
    # pprint((w.dot(p2) - 12) / w.norm())
    # pprint((w.dot(p3) - 12) / w.norm())
    # pprint((w.dot(p4) - 12) / w.norm())
    # pprint((w.dot(p5) - 12) / w.norm())
    # pprint(u.norm())
    # pprint(u.project(w))

    # pprint(u.dot(w))
    # pprint(m*n)
    # rref(m)
    # inverse()
    # determinant(m)
    # rotate_3d()

    # orthogonal_complement(m)
    # pprint(sp.Q.orthogonal(m))
    # pprint(m.columnspace(simplify=sp.fu))

    # q = inverse(m)
    # q = transpose(m)
    # pprint(q)

    # composition_txfms()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
