#!/usr/bin/python

"""This implements the exercises of minhash part for book: 'Mining the massive
dataset(v1.3)'. """

import sys
import itertools as itor
import math
import re
import numpy as np

def fig_3_2():
    """Return the data of Fig. 3.2(Column-major)"""
    
    #     a,b,c,d,e
    s1 = [1,0,0,1,0]
    s2 = [0,0,1,0,0]
    s3 = [0,1,0,1,1]
    s4 = [1,0,1,1,0]

    mat = [s1,s2,s3,s4]

#    num_rows, num_cols = len(mat[0]), len(mat)

    #data = np.array(num_rows, num_cols)
    #for i in num_cols:
        #for j in num_rows:
            #data[j,i] = mat[i][j]

    return mat

def fig_3_4():
    """Return the data of Fig. 3.4"""
    
    # The data is the same with Fig. 3.2
    return fig_3_2()

def fig_3_5():
    """Column-major"""

    s1 = [0,0,1,0,0,1]
    s2 = [1,1,0,0,0,0]
    s3 = [0,0,0,1,1,0]
    s4 = [1,0,1,0,1,0]

    return [s1,s2,s3,s4]
# For two sets s1, s2, rows divided into:
# Type X rows: 1 in both columns
# Type Y rows: 1 in one column, 0 in the other column
# Type Z rows: 0 in both columns

def shape(col_major):
    num_rows, num_cols = len(col_major[0]), len(col_major)
    return num_rows, num_cols

def jaccard_sim_naive(char_mat):
    """The naive method to calculate jaccard similarity of the characteristic
    matrix: char_mat"""
    
    _, set_num = shape(char_mat)  # num cols
    sim_mat = np.zeros((set_num, set_num))
    for i,s1 in enumerate(char_mat):
        for j,s2 in enumerate(char_mat):
            if i == j:
                sim_mat[i,j] = 1
            else:
                num_X, num_Y = 0, 0
                # Calculate number of type X and Y rows
                for v1, v2 in zip(s1, s2):
                    if v1 == v2 == 1:
                        num_X += 1
                    elif v1 != v2:
                        num_Y += 1
                sim_mat[i,j] = float(num_X) / float(num_X + num_Y)

    return sim_mat


def minhash_naive(char_mat):
    num_rows = len(char_mat[0])
    set_num = len(char_mat)
    sim_mat = np.zeros((set_num, set_num))
    total_num = math.factorial(num_rows)
    for i,s1 in enumerate(char_mat):
        for j,s2 in enumerate(char_mat):
            same_hash_num = 0
            for perm in itor.permutations(xrange(num_rows)):
                for idx in perm:
                    if s1[idx] == s2[idx] == 1:
                        # permutation that make the two columns hash
                        # to the same value
                        same_hash_num += 1
                        break
                    elif s1[idx] != s2[idx]:
                        # permuation that make the two colums hash to
                        # different value
                        break

            sim_mat[i,j] = float(same_hash_num) / float(total_num)
    
    return sim_mat

def calc_sig_mat(char_mat, hash_funcs, elements=None):
    """Calculate signature matrix from characteristic matrix"""
    num_rows, num_cols = len(char_mat[0]), len(char_mat)
    num_hf = len(hash_funcs)
    sig_mat = np.zeros((num_hf, num_cols))

    elements = elements or xrange(num_rows)


    for i in xrange(num_hf):
        for j in xrange(num_cols):
            sig_mat[i,j] = sys.maxint

    for k in xrange(num_rows):
        for i,hf in enumerate(hash_funcs):
            hash_value = hf(elements[k])
            for j,cols in enumerate(char_mat):
                if cols[k] == 1:
                    sig_mat[i,j] = min(sig_mat[i,j], hash_value)

    return sig_mat


def jaccard_sim(sig_mat):

    num_rows, num_cols = sig_mat.shape # num_hf X num_sets
    
    sim_mat = np.zeros((num_cols, num_cols))

    for i in xrange(num_cols):
        for j in xrange(num_cols):
            if i == j:
                sim_mat[i,j] = 0
            else:
                same_hash_num = 0
                for k in xrange(num_rows):
                    if sig_mat[k,i] == sig_mat[k,j]:
                        same_hash_num += 1
                sim_mat[i,j] = float(same_hash_num) / float(num_rows)

    return sim_mat

def exercise_3_3_1():
    """Exercise 3.3.1 : Verify the theorem from Section 3.3.3, which relates the Jac-
    card similarity to the probability of minhashing to equal values, for the partic-
    ular case of Fig. 3.2."""

    char_mat = fig_3_2()

    def a():
        """Compute the Jaccard similarity of each of the pairs of columns in Fig. 3.2."""
        print jaccard_sim_naive(char_mat)

    def b():
        """Compute, for each pair of columns of that figure, the fraction of the 120
        permutations of the rows that make the two columns hash to the same value."""
        print minhash_naive(char_mat)

    #a()
    #b()

def exercise_3_3_2():
    def h1(x):
        return (x+1)%5

    def h2(x):
        return (3*x+1)%5

    def h3(x):
        return 2*x+4

    def h4(x):
        return 3*x - 1

    #print calc_sig_mat(fig_3_4(), [h1,h2,h3,h4])

def exercise_3_3_3():
    char_mat = fig_3_5()

    hash_funcs = [lambda x: (2*x+1)%6, lambda x: (3*x+2)%6, lambda x: (5*x+2)%6]

    def a():
        """Compute the minhash signature for each column if we use the following
        three hash functions: h1 (x) = 2x + 1 mod 6; h2 (x) = 3x + 2 mod 6;
        h3 (x) = 5x + 2 mod 6."""

        print calc_sig_mat(char_mat, hash_funcs)

    def b():
        """Which of these hash functions are true permutations?"""
        for hf in hash_funcs:
            res = []
            for i in xrange(6):
                res.append(str(hf(i)))
            print ','.join(res)

        # Therefore, h3(x) is true permutation

    def c():
        print "True jaccard similarity:"
        print jaccard_sim_naive(char_mat)
        print "\nEsitmated jaccard similarity:"
        print jaccard_sim(calc_sig_mat(char_mat, hash_funcs))

    #a()
    #b()
    #c()

def exercise_3_3_4():
    pass

def exec_exercises(_globals):
    prog = re.compile('exercise_.*')
    for func_name in _globals:
        m = prog.match(func_name)
        if m:
            _globals[m.group(0)]()

exec_exercises(globals())
