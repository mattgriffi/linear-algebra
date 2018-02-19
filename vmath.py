"""This file defines some useful functions for performing calculations
with Vectors.
"""


import itertools

from vector import Vector


def gs(vectors, normal=True):
    """Performs the Gram-Schmidt process.

    Parameters
    ----------
    vectors : iterable
        Input iterable of Vectors. These Vectors may be linearly dependent and
        may include zero Vectors.
    normal : bool, optional
        If True, the output will be normalized. Default True.

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
    new = []
    for v in vectors:
        for p in new:
            v = v - v.project_onto(p)
        if v:  # Do not include zero vectors
            new.append(v)
        # Any more vectors would be linearly dependent, so stop
        if len(new) == v.dimension():
            break
    return normalize(new) if normal else new


def calculate_coefficients(orthonormal_basis, u):
    """Calculates the coefficients needed for a linear combination of the Vectors
    in orthonormal_basis to form the Vector u.

    Parameters
    ----------
    orthonormal_basis : iterable
        Input iterable of Vectors. These Vectors must form an orthonormal
        basis for a vector space containing u.
    u : Vector
        The Vector to construct via a linear combination.

    Returns
    -------
    list
        A list of the coefficients needed to construct u from orthonormal_basis.
        The coefficients will be in the same order as the Vectors in orthonormal_basis.

    Raises
    ------
    DimensionError
        If any of the input Vectors differ in dimension.

    Notes
    -----
    It must be possible to take the dot product of the Vectors in orthonormal_basis
    with u.
    """
    return [u.dot(v) for v in orthonormal_basis]


def linear_combination(basis, coefficients):
    """Performs a linear combination of basis using coefficients.

    Parameters
    ----------
    basis : iterable
        Input iterable of Vectors.
    coefficients : iterable
        Input iterable of scalars.

    Returns
    -------
    Vector
        A new Vector formed from the linear combination.

    Raises
    ------
    DimensionError
        If any of the input Vectors differ in dimension.
    """
    n = basis[0].dimension()
    return sum((k * v for k, v in zip(coefficients, basis)), Vector(zero=n))


def normalize(vectors):
    """Normalizes the given Vectors.

    Parameters
    ----------
    vectors : iterable
        Input iterable of Vectors.

    Returns
    -------
    list
        A new list of normalized Vectors.
    """
    return [v.normalize() for v in vectors]


def are_orthogonal(vectors):
    """Checks the given Vectors for orthogonality.

    Parameters
    ----------
    vectors : iterable
        Input iterable of Vectors.

    Returns
    -------
    bool
        True if all Vectors in vectors are orthogonal to each other, otherwise
        False. It must be possible to take the dot product of the Vectors.

    Raises
    ------
    DimensionError
        If any of the input Vectors differ in dimension.

    Notes
    -----
    This checks the dot product of every pair of Vectors in the input. As such,
    the time complexity is O(nv^2) where v is the number of vectors in the
    input and n is the dimension of the vectors.
    """
    for u, v in itertools.combinations(vectors, 2):
        if u.dot(v) != 0:
            return False
    return True


def are_normal(vectors):
    """Checks whether the given Vectors are normal.

    Parameters
    ----------
    vectors : iterable
        Input iterable of Vectors.

    Returns
    -------
    bool
        True if all Vectors in vectors are normal, otherwise False.

    Notes
    -----
    This compares the norm of each vector to 1 with a tolerance of 15 decimal
    places. It calls norm() on each Vector in vectors. As such, the time
    complexity is O(nv) where v is the number of vectors in the input and n is
    the dimension of the vectors.
    """
    for v in vectors:
        if not v.is_normal():
            return False
    return True


def are_orthonormal(vectors):
    """Checks whether the given Vectors are orthonormal.

    Parameters
    ----------
    vectors : iterable
        Input iterable of Vectors.

    Returns
    -------
    bool
        True if all Vectors in vectors are orthonormal, otherwise False.
    """
    return are_normal(vectors) and are_orthogonal(vectors)
