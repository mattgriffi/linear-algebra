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

    s = (u, v, w)

    m = Matrix(s)

    for col in m.columns:
        print(col)

    for row in m.rows:
        print(row)


if __name__ == "__main__":
    main()
