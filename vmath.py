"""This file defines some useful functions for performing calculations
with Vectors.
"""


import itertools
import math

from vector import Vector, check_dimensions


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
            check_dimensions(v, p, "perform the Gram-Schmidt process on")
            v = v - project(v, p)
        if v:  # Do not include zero vectors
            new.append(v)
        # Any more vectors would be linearly dependent, so stop
        if len(new) == v.dim:
            break
    return normalize_all(new) if normal else new


def dot(u, v):
        """Performs the standard dot product with the Vectors u and v,
        defined as:
        (u1)(v1) + (u2)(v2) + ... + (un)(vn)

        Parameters
        ----------
        u : Vector
        v : Vector

        Returns
        -------
        type: scalar
            The result of taking the dot product of u with v.
            Rounds to 15 decimal places.

        Raises
        ------
        DimensionError
            If the Vectors' dimensions differ.
        """
        check_dimensions(u, v, "dot")
        if not u or not v:  # <O, O> = <u, O> = <O, u> = 0
            return 0
        # Round the result to 15 decimal places, otherwise u.dot(v) == 0 can
        # be False for some orthogonal vectors
        return round(sum(i * j for i, j in zip(u, v)), 15)


def is_normal(u):
    """Determines whether u is normal.

    Parameters
    ----------
    u : Vector

    Returns
    -------
    bool
        True if u is normal, otherwise False.

    Notes
    -----
    This compares the norm of u to 1 with a tolerance of 15 decimal places.
    """
    return math.isclose(norm(u), 1, abs_tol=1e-15)


def norm(u):
    """Returns the Euclidean norm of the Vector u.
    """
    if not u:  # |O| = 0
        return 0
    return math.sqrt(norm2(u))


def norm2(u):
    """Returns the square of the Euclidean norm of the Vector u.
    """
    if not u:  # |O|^2 = 0
        return 0
    return sum(i * i for i in u)


def normalize(u):
    """Returns the normalization of the Vector u.
    """
    if not u:
        return u
    return u / norm(u)


def project(v, u):
    """Projects the Vector v onto u.

    Parameters
    ----------
    v : Vector
        The Vector to project.
    u : Vector
        The Vector to project onto

    Returns
    -------
    Vector
        The result of projecting v onto u.

    Raises
    ------
    DimensionError
        If the Vectors' dimensions differ.
    ValueError
        If u is the zero vector of the appropriate dimension.
    """
    check_dimensions(u, v, "project")
    if not u:
        raise ValueError("Cannot project onto a zero vector.")
    return dot(u, v) / norm2(u) * u


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
    return [dot(u, v) for v in orthonormal_basis]


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
    n = basis[0].dim
    return sum((k * v for k, v in zip(coefficients, basis)), Vector(zero=n))


def normalize_all(vectors):
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
    return [normalize(v) for v in vectors]


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
        if dot(u, v) != 0:
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
        if not is_normal(v):
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
