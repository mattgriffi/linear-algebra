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
        (3, 1, 1),
        (1, -3, -1)
    )

    B = [
        Vector(1, 1, 1),
        Vector(-1, 0, 1),
        Vector(0, 0, 1)
    ]
    
    C = [
        Vector(1, 2),
        Vector(-1, 1)
    ]

    A = mmath.get_transformation_matrix(T, B, C)
    u = Vector(1, 0, 2)

    printm(A * mmath.change_basis(u, B))

    for e in vmath.get_standard_unit_vectors(5):
        printm(e)


def printm(A):
    print(A)
    print()


if __name__ == "__main__":
    main()
