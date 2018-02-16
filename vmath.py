"""This file defines some useful functions for performing calculations
with Vectors.
"""


def gs(vectors):
    """Performs the Gram-Schmidt process.

    Parameters
    ----------
    vectors : iterable
        Input iterable of Vectors. These Vectors may be linearly dependent and
        may include zero vectors.

    Returns
    -------
    list
        A new list of linearly independent, orthonormal Vectors that span the
        same space as the input Vectors.

    Raises
    ------
    DimensionError
        If any of the input Vectors differ in dimension.
    """
    # TODO make sure the output is linearly independent
    new = []
    for v in vectors:
        for p in new:
            v = v - v.project_onto(p)
        if v:  # do not include zero vectors
            new.append(v)
    return normalize(new)


def normalize(vectors):
    """Normalizes the given Vectors.

    Parameters
    ----------
    vectors
        Input iterable of Vectors.

    Returns
    -------
    list
        A new list of normalized Vectors.
    """
    return [v.normalize() for v in vectors]
