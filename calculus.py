from sympy import init_printing, pprint
import numpy as np
# import matplotlib.pyplot as plt


def fn2(x):
    return (x ** 2 - 25) / (x + 5)


if __name__ == '__main__':
    init_printing(use_unicode=True)
    pprint(fn2(-5.0000001))