# 11.22
# Implementation of the barrier function

# TODO: Correct np.divide implementation

import matplotlib.pyplot as plt
import scipy
import numpy as np

def barrierOptimization(A, b, mu, maxiter, epsilon):
    Amax = np.max(A, 0);
    Amin = np.max(np.negative(A), 0);
    r = np.max(Amax * np.ones((2, 1)) + Amin * np.ones((2, 1)));
    u = (.5 / r) * np.ones((2, 1))
    l = -(.5 / r) * np.ones((2, 1))
    t = 1;
    # 1. Centering Step: computer x*(t)
    # Just calculate these once
    obj = u - l;
    barrier = b - Amax * u + Amin * l;
    obj_grad = -np.diagflat(np.divide(obj))
    obj_hess = -np.diagflat(np.)
    barrier_grad = -np.diagflat(np.divide(barrier))
    barrier_hess = -np.diagflat(np.linalg.pinv(np.linalg.pinv(barrier)))
    y = (t * - np.sum(np.log(obj))) - np.sum(np.log(barrier))
    grad = (t * [[-obj_grad], [obj_grad]]
            + [[Amax], [-Amin]] * barrier_grad)
    # hess = (t *
    

    # 2. Update

    # 3. Stopping Criterion

    # Increase t



# MAIN FUNCTION
if __name__ == '__main__':
    A = [[0, -1],
         [2, -4],
         [2,  1],
         [-4, 4],
         [-4, 0]];
    b = np.ones((2, 1));
    alpha = 0.2;
    beta = 0.5;
    mu = 20;  # Suggested everywhere in the text.
    maxiter = 100;
    epsilon = 1e-6;
    barrierOptimization(A, b, mu, maxiter, epsilon);

