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

    Q, R = mmath.factorize(A)

    r = math.sqrt

    Qp = 1 / r(6) * Matrix(
        Vector(r(2), 1, r(3)),
        Vector(r(2), 1, -r(3)),
        Vector(r(2), -2, 0)
    )

    Rp = 1 / r(6) * Matrix(
        Vector(3 * r(2), 2 * r(2), 2 * r(2)),
        Vector(0, 2, 2),
        Vector(0, 0, 2 * r(3))
    )

    print(Q)
    print()
    print(Qp)
    print()
    print(R)
    print()
    print(Rp)


if __name__ == "__main__":
    main()
