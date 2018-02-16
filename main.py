"""
This program allows for various linear algebra calculations.
"""


import vmath

from vector import Vector


def main():
    u = Vector(1, 0, 1)
    v = Vector(3, 1, 1)
    w = Vector(-1, -1, -1)
    x = Vector(0, 0, 0)
    y = Vector(0, 0, 1)

    print(bool(x), bool(y))

    b = [u, v, w]

    print(-u)

    for vector in vmath.gs(b):
        print(vector)


if __name__ == "__main__":
    main()
