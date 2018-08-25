# -*- coding: utf-8 -*-
# conjugate gradient needed for model based RL algorithms, e.g, TRPO,PPO

import numpy as np
from scipy.sparse.linalg import cg
import tensorflow as tf
import time


def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.

    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point

    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in xrange(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print 'Itr:', i
            break
        p = beta * p - r
    return x

# x_ph = tf.placeholder('float32', [None, None])
# r = tf.matmul(A, x_ph) - b

if __name__ == '__main__':
    n = 1000
    P = np.random.normal(size=[n, n])
    A = np.dot(P.T, P)
    b = np.ones(n)

    t1 = time.time()
    print 'start'
    x = conjugate_grad(A, b)
    t2 = time.time()
    print t2 - t1
    x2 = np.linalg.solve(A, b)
    t3 = time.time()
    print t3 - t2
    x3 = cg(A, b)
    t4 = time.time()
    print t4 - t3

    # print np.dot(A, x) - b
