"""
This program allows for various linear algebra calculations.
"""


import cProfile
import math
import pstats

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    A = Matrix(
        (1, 4, 5, 3),
        (5, 22, 27, 11),
        (6, 19, 27, 31),
        (5, 28, 35, -8)
    )
    b = Vector(1, 2, 3, 4)

    cProfile.runctx('solve1(A, b)', {'solve1': solve1}, {'A': A, 'b': b}, sort='tottime')
    cProfile.runctx('solve2(A, b)', {'solve2': solve2}, {'A': A, 'b': b}, sort='tottime')


def solve1(A, b):
    for _ in range(1000):
        aug = mmath.augment(A, b)
        R = mmath.rref(aug)
        _, x = mmath.deaugment(R, 1)


def solve2(A, b):
    for _ in range(1000):
        L, U = mmath.factor_LU(A)
        y = mmath.deaugment(mmath.rref(mmath.augment(L, b)), 1)[1]
        x = mmath.deaugment(mmath.rref(mmath.augment(U, y)), 1)[1]


def printm(A):
    print(A)
    print()


def printf(A):
    print(A.str_fractions())
    print()


if __name__ == "__main__":
    main()
