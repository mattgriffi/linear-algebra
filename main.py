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
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2)
    )
    I = Matrix(identity=3)

    for val, vecs in mmath.eig(A, n=100):
        printe(val, vecs)
    
    B = A + -4*I
    printm(mmath.rref(B))


def printe(eigenvalue, eigenvectors):
    x = "\t|\t".join(v.str_fractions() for v in eigenvectors)
    print("eigenvalue: ", eigenvalue, " eigenvectors: ", x)


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
