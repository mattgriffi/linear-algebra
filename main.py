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
        (1, 3, 4, 5),
        (2, 6, -8, -6)
    )

    for v in mmath.row_space(A):
        print(v)
    for v in mmath.column_space(A):
        print(v)
    print(mmath.rank(A))
    print(mmath.nullity(A))


def printm(A):
    print(A)
    print()


if __name__ == "__main__":
    main()
