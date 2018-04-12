"""This file defines some useful functions for performing calculations
with Matrices.
"""

import functools
import itertools
import math
import operator

import vmath
from matrix import Matrix, DimensionError
from vector import Vector


class FactorizationError(Exception):
    pass


def _check_square(error_message):
    """Checks the first argument of the decorated function to
    see if it is a square Matrix. If not, raises DimensionError
    with the given error_message.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            A = args[0]
            if not isinstance(A, Matrix):
                raise ValueError("First arg of function must be Matrix.")
            if A.dim.rows != A.dim.columns:
                raise DimensionError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def transpose(A):
    """Calculates the transpose of Matrix A.
    """
    return Matrix(A.columns)


def trace(A):
    """Calculates the sum of the leading diagonal of Matrix A.
    """
    return sum(A[i][i] for i in range(min(A.dim)))


def rank(A, is_rref=False):
    """Returns the rank of Matrix A.
    """
    return len(row_space(A, is_rref))


def nullity(A, is_rref=False):
    """Returns the nullity of Matrix A.
    """
    return A.dim.columns - rank(A, is_rref)


def solve(A, b=None):
    """Solves the given linear system.

    Parameters
    ----------
    A : Matrix
        The coefficient Matrix of the system.
    b : Vector, default None
        The constant Vector. If None, the system is
        assumed to be homogeneous.

    Returns
    -------
    list of Vectors
        The solution set of Ax = b.
    boolean
        Whether or not there is a unique solution.
    """
    if b is None:
        b = Vector(zero=A.dim.rows)
    R, x = deaugment(rref(augment(A, b)), 1)
    x = list(x)
    m, n = R.dim
    unique = False

    # free_variables holds copies of x with a single free
    # variable set to 1; one copy for each free variable
    free_variables = []
    basis = []
    good_rows = set()

    # Columns without a leading 1 are free variables,
    # so set the corresponding value in x to 1
    for c in range(n):
        r = m - 1
        # Skip through zeroes at bottom of column
        while r >= 0 and R[r][c] == 0:
            r -= 1
        # Track which rows have a leading 1
        if r >= 0 and R[r][c] == 1 and r not in good_rows:
            good_rows.add(r)
            print(R[r])
        elif r < 0 or r in good_rows:
            print(R[r])
            y = x[:]
            y[c] = 1
            free_variables.append(y)
    
    # There were no free variables
    if not free_variables:
        free_variables.append(x)
        unique = True

    # Build the basis vectors
    for y in free_variables:
        for r in range(m - 1, -1, -1):
            # Skip 0 rows
            if not R[r]:
                continue
            c = 0
            # Get to the leading 1
            while R[r][c] != 1:
                c += 1
            i = c  # i is the index of the variable for this row
            c += 1  # Skip the leading 1
            # Use back substitution
            while c < n:
                if R[r][c] != 0:
                    y[i] -= R[r][c] * y[c]
                c += 1
        basis.append(Vector(y))
    return basis, unique
        

@_check_square("Cannot invert non-square Matrix.")
def invert(A):
    """Returns the inverse of Matrix A. Raises Dimension error if A
    is not square. Uses Gauss-Jordan Elimination.
    """
    A_augment_I = augment(A, Matrix(identity=A.dim.columns))
    _, A_inverse = deaugment(rref(A_augment_I), A.dim.columns)
    return A_inverse


@_check_square("Cannot invert non-square Matrix.")
def invert2(A):
    """Returns the inverse of Matrix A. Raises Dimension error if A
    is not square. Uses the determinant and adjoint of A.
    """
    return adjoint(A) / det(A)


@_check_square("Cannot find adjoint of non-square Matrix.")
def adjoint(A):
    return transpose(cofactor(A))


@_check_square("Cannot find determinant of non-square Matrix.")
def det(A):
    """Returns the determinant of Matrix A.
    """
    if A.dim.rows == A.dim.columns == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    # Expand the first row to calculate determinant
    return sum(A[0][c] * cofactor_element(A, 0, c) for c in range(A.dim.columns))


@_check_square("Cannot find determinant of non-square Matrix.")
def det_triangular(A):
    """Returns the determinant of triangular Matrix A. This is more
    efficient than the general det function.
    """
    diag = (A[i][i] for i in range(A.dim.rows))
    return functools.reduce(operator.mul, diag)


@_check_square("Cannot find cofactor of non-square Matrix.")
def cofactor(A):
    """Returns the cofactor Matrix of Matrix A.
    """
    rows = [[cofactor_element(A, r, c) for c in range(A.dim.columns)]
            for r in range(A.dim.rows)]
    return Matrix(rows)


@_check_square("Cannot find cofactor in non-square Matrix.")
def cofactor_element(A, r, c):
    """Returns the cofactor of the entry in Matrix A.
    """
    return math.pow(-1, r + c) * minor(A, r, c)


@_check_square("Cannot find minor in non-square Matrix.")
def minor(A, r, c):
    """Returns the minor the entry in Matrix A.
    """
    rows = list(list(row) for row in A.rows)
    del rows[r]
    for row in rows:
        del row[c]
    m = Matrix(rows)
    return det(m)


@_check_square("Cannot exponentiate non-square Matrix.")
def power(A, n):
    """Returns square Matrix A raised to the n'th power.
    """
    def r(B, n):
        # Use exponentiation by squaring method
        if n <= 1:
            return B
        elif n % 2:  # Odd exponent
            return B * r(B * B, (n - 1) / 2)
        else:  # Even exponent
            return r(B * B, n / 2)

    return r(A, n)


def augment(A, B):
    """Augments Matrix A with Matrix or Vector B.

    Parameters
    ----------
    A : Matrix
        The Matrix to augment with B.
    B : Matrix or Vector
        The Matrix or Vector to attach to A.

    Returns
    -------
    Matrix
        The result of augmenting A with B.

    Raises
    ------
    DimensionError
        If A and B do not have the same number of rows (Matrix), or if
        B does not have as many elements as A has rows (Vector).
    """
    if isinstance(B, Vector):
        if A.dim.rows != B.dim:
            raise DimensionError("Cannot augment Matrix with Vector of innapropriate dimension.")
        return Matrix(*A.columns, B, columns=True)
    else:
        if A.dim.rows != B.dim.rows:
            raise DimensionError("Cannot augment Matrix with Matrix of innapropriate dimension.")
        return Matrix(*A.columns, *B.columns, columns=True)


def deaugment(A, n):
    """Splits the last n columns off of A. Returns the resulting Matrices or
    Matrix and Vector if n is 1.
    """
    if n == 1:
        return Matrix(A.columns[:-1], columns=True), A.columns[-1]
    return Matrix(A.columns[:-n], columns=True), Matrix(A.columns[-n:], columns=True)


@_check_square("Cannot factor non-square Matrix.")
def factor_QR(A):
    """Performs QR factorization on invertible Matrix A. Returns (Q, R).
    """
    # Use Gram-Schidt to calculate orthogonal Matrix Q from columns of A
    Q = Matrix(vmath.gs(A.columns), columns=True)

    # If A is not invertible, vmath.gs will not return a square matrix
    if A.dim != Q.dim:
        raise DimensionError("Matrix A must be invertible.")

    # Upper triangular Matrix R = (Q^T)A
    R = transpose(Q) * A

    return Q, R


@_check_square("Cannot calculate eigenvalues of non-square Matrix")
def eigval(A, n=10, precision=4):
    """Calculates the eigenvalues of square Matrix A using n iterations
    of the QR algorithm. Returns a Vector of the eigenvalues.
    """
    for _ in range(n):
        Q, R = factor_QR(A)
        A = R * Q
    return Vector(round(A[i][i], precision) for i in range(min(A.dim)))


@_check_square("Cannot calculate eigenvectors of non-square Matrix")
def eigvec(A, eigenvalue):
    """Calculates the eigenvectors for a given eigenvalue of A. Returns
    a list of Vectors.
    """
    solution, unique = solve(A - eigenvalue * Matrix(identity=A.dim.rows))
    return solution


@_check_square("Cannot calculate eigenvalues of non-square Matrix")
def eig(A, n=10, precision=4):
    """Calculates the eigenvalues and eigenvectors of square Matrix A.
    Returns a generator of tuples: (λ, [u1, u2, ...]). Eigenvalues
    are calculated with n iteration of the QR algorithm and rounded to
    precision decimal places.
    """
    for eigenvalue in eigval(A, n, precision):
        yield eigenvalue, eigvec(A, eigenvalue)


def poly(eigenvalues):
    """Returns a Vector representing the coefficients of the
    characteristic polynomial for the given eigenvalues.
    For p(λ) = (c_n)(λ^n) + (c_n-1)(λ^n-1) + ... + (c_1)λ + c_0
    Returns Vector(c_0, c_1, ..., c_n-1, c_n)
    """
    n = len(eigenvalues)

    def gen():
        for i in range(n):
            combinations = itertools.combinations(eigenvalues, n - i)
            products = (functools.reduce(operator.mul, c) for c in combinations)
            yield sum(products)
        yield -1 if n % 2 else 1

    return Vector(gen())


@_check_square("Cannot factor non-square Matrix.")
def factor_LU(A):
    """Performs LR factorization on invertible Matrix A. Returns (L, U).
    """
    m, n = A.dim
    r, c = 0, 0
    zero = lambda x: math.isclose(x, 0, abs_tol=1e-15)
    L = list(list(row) for row in Matrix(identity=n))
    U = A

    # Make U into an upper triangular Matrix while performing
    # the opposite operations on L
    while r < m and c < n:
        # If the diagonal entry is 0, we're stuck.
        if zero(U[r][c]):
            raise FactorizationError("Matrix cannot be factored.")

        L[r][c] = U[r][c]
        # Make diagonal entry 1
        U = row_multiply(U, r, 1 / U[r][c])
        # Eliminate the rest
        for i in range(r + 1, m):
            if not zero(A[i][c]):
                L[i][c] = U[i][c]
                U = row_add_mul(U, r, i, -1 * U[i][c])
        r += 1
        c += 1

    return Matrix(L), U


def rref(A):
    """Performs Gauss-Jordan Elimination, returns the reduced row
    echelon form of A.
    """
    # This will be very inefficient with immutable matrices and vectors
    m, n = A.dim
    r, c = 0, 0
    zero = lambda x: math.isclose(x, 0, abs_tol=1e-15)

    while r < m and c < n:
        try:
            # Get the first unprocessed row index of a nonzero element in column c
            nz = next(i for i, x in enumerate(A.columns[c][r:]) if not zero(x)) + r
            # Swap that row with row r
            A = row_swap(A, r, nz)
        except StopIteration:
            # If there are no nonzero elements left, skip column
            c += 1
            continue

        # Divide row r by its leading element so it becomes 1
        assert not zero(A[r][c])
        A = row_multiply(A, r, 1 / A[r][c])

        # Eliminate nonzero elements of all other rows in column c
        for i in (x for x in range(m) if x != r):
            if not zero(A[i][c]):
                A = row_add_mul(A, r, i, -1 * A[i][c])

        # Go to next row and column
        r += 1
        c += 1

    return A


def get_transformation_matrix(T, B, C):
    """Returns the transformation Matrix for T with respect to domain
    basis B and codomain basis C.
    """
    W = Matrix(C, columns=True)
    result_columns = (deaugment(rref(augment(W, v)), 1)[1] for v in transform_all(T, B))
    return Matrix(result_columns, columns=True)


def change_basis(v, B):
    """Returns Vector v in terms of basis Vector set B.
    """
    A = Matrix(*B, v, columns=True)
    _, vb = deaugment(rref(A), 1)
    return vb


def transform_all(A, V):
    """Returns an iterator that applies Matrix A to each Vector in V.
    """
    for v in V:
        yield A * v


def compose(M):
    """Returns the resulting Matrix from multiplying the Matrices in
    iterable M together. The result is M[0] * M[1] * ... * M[len(M) - 1]
    """
    return functools.reduce(operator.mul, M)


def row_space(A, is_rref=False):
    """Returns an iterator of Vectors forming a basis for the row space of Matrix A.
    """
    R = A if is_rref else rref(A)
    for row in R:
        if row:
            yield row


def column_space(A):
    """Returns an iterator of Vectors forming a basis for the column space
    of Matrix A.
    """
    return row_space(transpose(A))


def row_swap(A, row1, row2):
    """Swaps row1 with row2 and returns a new Matrix.
    """
    if row1 == row2:
        return A
    rows = list(A.rows)
    rows[row1], rows[row2] = rows[row2], rows[row1]
    return Matrix(rows)


def row_add(A, row1, row2):
    """Adds row1 to row2 and returns a new Matrix.
    """
    if not row1:
        return
    rows = list(A.rows)
    rows[row2] = rows[row1] + rows[row2]
    return Matrix(rows)


def row_multiply(A, row, k):
    """Multiplies row by scalar k and returns a new Matrix.
    """
    if k == 1:
        return A
    rows = list(A.rows)
    rows[row] = k * rows[row]
    return Matrix(rows)


def row_add_mul(A, row1, row2, k):
    """Adds a multiple of row1 to row2 and returns a new Matrix.
    """
    if k == 0:
        return A
    rows = list(A.rows)
    rows[row2] = k * rows[row1] + rows[row2]
    return Matrix(rows)
