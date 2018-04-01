"""
This program allows for various linear algebra calculations.
"""


import math

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    T = Matrix(
        (1/3, 2/3),
        (2/3, 1/3)
    )
    x = Vector(1, 0)

    for i in range(1, 5):
        printm(mmath.power(T, i) * x)
    
    printm(mmath.power(T, 100) * x)


def printm(A):
    print(A)
    print()


def printf(A):
    print(A.str_fractions())
    print()


if __name__ == "__main__":
    main()
