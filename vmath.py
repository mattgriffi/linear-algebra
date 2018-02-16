"""
This file defines some useful functions for performing calculations
with vectors.
"""


def gs(vectors):
    """Performs the Gram-Schmidt process and returns a new orthonormal
    list of vectors."""
    new = []
    for v in vectors:
        for p in new:
            v = v - v.project_onto(p)
        if v:  # do not include zero vectors
            new.append(v)
    return normalize(new)


def normalize(vectors):
    """Returns a new normalized list of vectors."""
    return [v.normalize() for v in vectors]
