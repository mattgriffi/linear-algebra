"""
This program allows for various linear algebra calculations.
"""


import math

import vmath
from matrix import Matrix
from vector import Vector


def main():
    u = Vector(2, 2, 2)
    v = Vector(-1, 0, -1)
    w = Vector(-1, 2, 3)
    x = Vector(1, 1, 1)

    s = (u, v, w)
    z = (x, x, x)

    A = Matrix(s)
    B = Matrix(z)
    C = B + A

    for col in C.columns:
        print(col)

    for row in C.rows:
        print(row)


if __name__ == "__main__":
    main()
