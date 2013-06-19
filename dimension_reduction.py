#!/usr/bin/python

# chapter 11

import math
import itertools
import numpy as np
def take(n, iterable):
    return list(itertools.islice(iterable, n))

EPSILON = 0.00001


def simple_M():
    return np.array([[1, 2],
                     [2, 1],
                     [3, 4],
                     [4, 3]])
def perfect_M():
    M = [[1, 1, 1, 0, 0],
         [3, 3, 3, 0, 0],
         [4, 4, 4, 0, 0],
         [5, 5, 5, 0, 0],
         [0, 0, 0, 4, 4],
         [0, 0, 0, 5, 5],
         [0, 0, 0, 2, 2]]
    return np.array(M)

def perfect_data():
    U = np.array([[.14, .42, .56, .70, 0, 0, 0],
                  [0, 0, 0, 0, .60, .75, .30]]).T
    VT = np.array([[.58, .58, .58, 0, 0],
                   [0, 0, 0, .71, .71]])
    Sigma = np.array([[12.4, 0,],
                      [0, 9.5]])
    return U, Sigma, VT

def noise_data():
    U = np.array([[0.13, 0.41, 0.55, 0.68, 0.15, 0.07],
                  [0.02, 0.07, .09, .11, -.59, -.73, -.29]]).T
    VT = np.array([[0.56, 0.59, 0.56, 0.09, 0.09],
                  [0.12, -0.02, 0.12, -0.69, -0.69]])
    Sigma = np.array([[12.4, 0],
                      [0, 0.5]])

    return U, Sigma, VT

def demo_querying_using_concepts():
    # q represents a user's rating of movies
    q = np.array([4, 0, 0, 0, 0])
    _, _, VT = perfect_data()

    # map q to concept space.
    concept = np.dot(q, VT.T)
    print concept
    # map back 
    print np.dot(concept, VT)


def frobenius_norm_square(V):
    return np.sum(np.square(V))

def frobenius_norm(V):
    return math.sqrt(frobenius_norm_square(V))

def power_iteration(M, max_loop=100, power_iteration_epsilon=EPSILON):
    n,m = M.shape
    x = np.ones(n)

    # in case Mx = 0
    if np.sum(np.abs(np.dot(M, x))) <= power_iteration_epsilon:
        x[0] = -1.0

    while max_loop >= 1:
        x_k = np.dot(M, x)
        x_k = x_k / (frobenius_norm(x_k) * 1.0)
        if frobenius_norm(x_k - x) < power_iteration_epsilon:
            x = x_k
            break
        x = x_k
        max_loop -=1

    eigen_vector = x
    eigen_value = x.T.dot(M).dot(x)
    return eigen_vector, eigen_value

def pseudoinverse(M_orig, diag=True, eps=EPSILON):
    M = M_orig.copy()
    if diag:
        return np.diag([x if x <= eps else 1.0/x for x in M.diagonal()])
    else:
        raise NotImplementedError


def eigen_solver(M_orig, min_eigvalue=0.01, *args, **kw):
    M = M_orig.copy()
    E = []
    Sigma = []
    while 1:
        eigvec, eigvalue = power_iteration(M, *args, **kw)
        if abs(eigvalue) <= min_eigvalue:
            break
        if eigvec[0] < 0:
            # modify the direction.
            eigvec = eigvec * (-1)
        E.append(eigvec)
        Sigma.append(eigvalue)
        M = M - eigvalue * np.outer(eigvec, eigvec)

    return np.array(E).T, np.diag(Sigma)


def pca(M, *args, **kw):
    tmp = M.T.dot(M)
    E, Sigma = eigen_solver(tmp, *args, **kw)
    return E, Sigma

def svd(M, *args, **kw):
    tmp = M.T.dot(M)
    V, SigmaSquare = eigen_solver(tmp, *args, **kw)
    U, SigmaSquare = eigen_solver(tmp.T, *args, **kw)
    return U, np.sqrt(SigmaSquare), V.T


RANDOMLY = 0
RANDOM_BY_PROB = 1
MAX_BY_PROB = 2

def _select_randomly(M, r):
    raise NotImplementedError

def _select_randomly_by_prob(M, r):
    raise NotImplementedError


def _calc_prob(v, fnorm):
    return 1.0 * frobenius_norm_square(v) / fnorm


def _select_max_by_prob(M, r):
    fnorm = frobenius_norm_square(M)
    # columns 
    column_probs = np.apply_along_axis(_calc_prob, 0, M, fnorm)
    # rows
    row_probs = np.apply_along_axis(_calc_prob, 1, M, fnorm)
    column_indices = [offset for offset,_ in take(r,
                                 sorted(enumerate(column_probs),
                                        key=lambda o: o[1],
                                        reverse=True))]
    row_indices = [offset for offset,_ in take(r,
                                               sorted(enumerate(row_probs),
                                                      key=lambda o: o[1],
                                                      reverse=True))]
    W = np.array([M[i,j] for i in reversed(row_indices)\
                  for j in reversed(column_indices)]).reshape(r,r)
    # The following code actually do the same thing using itertools.
    #W = M[zip(*itertools.product(reversed(row_indices),
                                 #reversed(column_indices)))].reshape(r,r)
    
    MT = M.T.copy()
    CT = []
    for column_indice in column_indices:
        prob = column_probs[column_indice]
        expected_value = math.sqrt(r * prob)
        CT.append(np.divide(MT[column_indice], expected_value))

    C = np.array(CT).T

    R = []
    for row_indice in row_indices:
        prob = row_probs[row_indice]
        expected_value = math.sqrt(r * prob)
        R.append(np.divide(M[row_indice], expected_value))

    R = np.array(R)
    return C, R, W

def cur(M, r, select_method=MAX_BY_PROB, *args, **kw):
    """CUR decomposition
    M is the matrix to be decomposed, r is the estimated rankd of
    the matrix"""

    def _select_cr():
        return {
            RANDOMLY: _select_randomly,
            RANDOM_BY_PROB: _select_randomly_by_prob,
            MAX_BY_PROB: _select_max_by_prob,
        }[select_method](M, r)
    
    C, R, W = _select_cr()
    X, Sigma, YT = svd(W)
    Sigma = np.square(pseudoinverse(Sigma))
    U = YT.T.dot(Sigma).dot(X.T)
    return C, U, R

def test_eigen_solver():
    arr = np.array([[3, 2],
                    [2, 6]])
    print eigen_solver(arr)
    arr = np.array([[.8, .3],
                    [.2, .7]])
    print eigen_solver(arr)

def test_svd():
    arr = np.array([[1, 1, 1, 0, 0],
                    [3, 3, 3, 0, 0],
                    [4, 4, 4, 0, 0],
                    [5, 5, 5, 0, 0],
                    [0, 2, 0, 4, 4],
                    [0, 0, 0, 5, 5],
                    [0, 1, 0, 2, 2]])
    U, Sigma, VT = svd(arr)
    print U
    print "***************"
    print Sigma
    print "***************"
    print VT

def test_select_max_by_prob():
    M = np.arange(0, 16).reshape(4,4)
    print M
    C, R, W = _select_max_by_prob(M, 2)
    print "***************"
    print C.shape
    print C
    print "***************"
    print R
    print R.shape
    print "***************"
    print W.shape
    print W

def test_cur():
    C, U, R = cur(perfect_M(), 4) 
    print C
    print U
    print R

def test_pca():
    M = simple_M()
    E, Sigma = pca(M)
    print E
    print "***************"
    print Sigma
    print "***************"
    print M.dot(E)

#demo_querying_using_concepts()
#test_eigen_solver()
#test_svd()
#test_select_max_by_prob()
#test_cur()
#test_pca()
#test_power_iteration()
