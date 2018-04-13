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
        (2, 2),
        (1, 3)
    )
    P, D = mmath.factor_PD(A, n=100)
    print("P:")
    printf(P)
    print("D:")
    printf(D)
    print("P * D * P^-1")
    printf(P * D * mmath.invert(P))


def printe(eigenvalue, eigenvectors):
    x = "\n".join(v.str_fractions() for v in eigenvectors)
    print("λ: ", eigenvalue, " u:\n", x)
    print()


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
