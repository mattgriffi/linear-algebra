"""
This program allows for various linear algebra calculations.
"""


import math

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    A = Matrix(
        Vector(1, 0, 4),
        Vector(1, 1, 6),
        Vector(-3, 0, -10)
    )

    I = Matrix(identity=3)
    AI = mmath.augment(A, I)
    Ap, Ip = mmath.deaugment(AI, 1)

    printm(mmath.rref(AI))


def printm(A):
    print(A)
    print()


if __name__ == "__main__":
    main()
