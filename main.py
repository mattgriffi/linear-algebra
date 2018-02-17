"""
This program allows for various linear algebra calculations.
"""


import vmath

from vector import Vector


def main():
    u = Vector(3, 0, 4, 2)
    v = Vector(1, 2, 7, 0)
    w = Vector(3, 2, -2, -5)
    x = 2*w

    b = [u, v, w, x]

    bp = vmath.gs(b)

    for vector in bp:
        print(vector)

    print(vmath.are_orthogonal(bp))
    print(vmath.are_orthogonal(b))


if __name__ == "__main__":
    main()
