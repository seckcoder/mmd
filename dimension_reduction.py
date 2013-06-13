#!/usr/bin/python

# chapter 11

import math
import numpy as np

EPSILON = 0.00001

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

def eigen_solver(M, min_eigvalue=0.01, *args, **kw):
    """Note M will be modified"""
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

def svd(M, *args, **kw):
    tmp = M.T.dot(M)
    V, SigmaSquare = eigen_solver(tmp, *args, **kw)
    tmp = M.dot(M.T)
    U, SigmaSquare = eigen_solver(tmp, *args, **kw)
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


def kbig(vec, k):
    if k <= 0:
        return None

    for offset,value in enumerate(vec):
        pass

def _select_max_by_prob(M, r):
    fnorm = frobenius_norm_square(M)
    # columns 
    column_probs = np.apply_along_axis(_calc_prob, 0, M, fnorm)
    # rows
    row_probs = np.apply_along_axis(_calc_prob, 1, M, fnorm)
    

def cur(M, r, select_method=MAX_BY_PROB, *args, **kw):
    """CUR decomposition
    M is the matrix to be decomposed, r is the estimated rankd of
    the matrix"""
    def select_cr():
        if select_method=


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

#demo_querying_using_concepts()
#test_eigen_solver()
test_svd()
