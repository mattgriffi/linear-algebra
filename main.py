"""This program allows for various linear algebra calculations.
"""


import cProfile
import math
import pstats

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    A = Matrix(
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    )
    b = Vector(0, 0, 0)

    R = mmath.rref(mmath.augment(A, b))
    printf(R)
    solution, unique = mmath.solve(A, b)
    print(unique)
    for v in solution:
        printf(v)


def printm(A):
    print(A)
    print()


def printf(A):
    print(A.str_fractions())
    print()


def printp(poly):
    print("p(λ) = ", end="")
    s = "{:+.2g}{}{}{}"
    terms = []
    for i, e in enumerate(poly):
        lam = "λ" if i > 0 else ""
        power = "^" if i > 1 else ""
        exp = i if i > 1 else ""
        terms.append(s.format(e, lam, power, exp))
    print(" ".join(terms))


if __name__ == "__main__":
    main()
