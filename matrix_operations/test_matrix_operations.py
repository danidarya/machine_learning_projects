import pytest
import numpy as np
from main import Matrix

a_elements = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]
b_elements = [[9, 8, 7, 6], [6, 5, 4, 3], [3, 2, 1, 0]]
m_elements = [[1, 2, 3], [5, 6, 7], [8, 5, 4]]
n_elements = [[5, 6, 7], [2, 6, 4], [8, 7, 5]]
y_elements = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
A = Matrix(3, 4, a_elements)
B = Matrix(3, 4, b_elements)
M = Matrix(3, 3, m_elements)
N = Matrix(3, 3, n_elements)
Y = Matrix(3, 3, y_elements)
numpy_A = np.array(a_elements)
numpy_B = np.array(b_elements)
numpy_M = np.array(m_elements)
numpy_N = np.array(n_elements)


def test_add():
    C = A.add(B)
    assert C.n_rows == 3
    assert C.n_cols == 4
    assert np.allclose(C.elements, numpy_A + numpy_B)


def test_wrong_add():
    with pytest.raises(IOError):
        A.add(M)


def test_subtract():
    C = B.subtract(A)
    assert C.n_rows == 3
    assert C.n_cols == 4
    assert np.allclose(C.elements, numpy_B - numpy_A)


def test_wrong_subtract():
    with pytest.raises(IOError):
        A.subtract(M)


def test_traspose():
    C = A.transpose()
    assert np.allclose(C.elements, numpy_A.T)


def test_multiply():
    C = A.multiply(B.transpose())
    assert C.n_rows == 3
    assert C.n_cols == 3
    assert np.allclose(C.elements, numpy_A.dot(numpy_B.T))
    C = B.multiply(A.transpose())
    assert np.allclose(C.elements, numpy_B.dot(numpy_A.T))


def test_wrong_multiply():
    with pytest.raises(IOError):
        A.multiply(B)


def test_det():
    d = M.det()
    assert np.allclose(d, np.linalg.det(numpy_M))


def test_wrong_det():
    with pytest.raises(IOError):
        d = A.det()


def test_inv():
    M_inv = M.inv()
    assert np.allclose(M_inv.elements, np.linalg.inv(numpy_M))


def test_wrong_inv():
    with pytest.raises(IOError):
        X = Y.inv()


def test_division():
    C = N.division(M)
    assert np.allclose(C.elements, numpy_N.dot(np.linalg.inv(numpy_M)))
