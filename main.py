"""
This program allows for various linear algebra calculations.
"""


import math

import vmath
from matrix import Matrix
from vector import Vector


def main():
    u = Vector(1, 1, -3)
    v = Vector(0, 1, 0)
    w = Vector(4, 6, -10)

    ui = Vector(-5, -4, 3/2)
    vi = Vector(0, 1, 0)
    wi = Vector(-2, -1, 1/2)

    s = (u, v, w)
    si = (ui, vi, wi)

    Ai = Matrix(si)
    A = Matrix(s)
    I = Matrix(
        Vector(1, 0, 0),
        Vector(0, 1, 0),
        Vector(0, 0, 1)
    )

    C = Ai * A

    for c in C.rows:
        print(c)


if __name__ == "__main__":
    main()
