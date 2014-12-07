# 11.22
# Implementation of the barrier function

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
    y = (t * - np.sum(np.log(obj))) - np.sum(np.log(barrier))
    grad = ((t * (-np.diagflat(np.pinv(obj)) + np.diagflat(np.pinv(obj))))
            + Amax * np.diagflat(np.pinv(barrier))
            - Amin * np.diagflat(np.pinv(barrier)))
    hess = ((t * (-np.diagflat(np.pinv(obj)) + np.diagflat(np.pinv(obj))))
            + Amax * np.diagflat(np.pinv(barrier))
            - Amin * np.diagflat(np.pinv(barrier)))
    
    print y
    

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

