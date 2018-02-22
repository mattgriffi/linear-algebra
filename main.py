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
        Vector(1, 1, 2),
        Vector(1, 1, 0),
        Vector(1, 0, 0)
    )

    I = Vector(-1, -2, -3)
    AI = mmath.augment(A, I)
    Ap, Ip = mmath.deaugment(AI, 1)

    print(AI)
    print()
    print(Ap)
    print()
    print(Ip)


if __name__ == "__main__":
    main()
