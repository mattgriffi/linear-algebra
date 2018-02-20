"""
This program allows for various linear algebra calculations.
"""


import math

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    u = Vector(1, 1, -3)
    v = Vector(0, 1, 0)
    w = Vector(4, 6, -10)
    x = Vector(1, 2, 3)

    ui = Vector(-5, -4, 3/2)
    vi = Vector(0, 1, 0)
    wi = Vector(-2, -1, 1/2)

    s = (u, v, w, x)
    si = (ui, vi, wi)

    Ai = Matrix(si)
    A = Matrix(s)
    I = Matrix(
        Vector(1, 0, 0),
        Vector(0, 1, 0),
        Vector(0, 0, 1)
    )

    H = Matrix(
        Vector(0, 0, 2, 4, 4),
        Vector(0, 3, 5, 3, 0),
        transpose=True
    )

    A = Matrix(
        Vector(1, 1),
        Vector(-1, 1),
        transpose=True
    )

    AH = A * H

    print(A)
    print()
    print(H)
    print()
    print(AH)


if __name__ == "__main__":
    main()
