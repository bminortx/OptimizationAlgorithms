# 11.22
# Implementation of the barrier function

# TODO: Correct np.true_divide implementation

import matplotlib.pyplot as plt
import numpy as np

def barrierOptimization(A, b, m, n, mu, maxiter, epsilon, eps_t):
    Amax = np.array(np.max(A, 0));
    Amin = np.array(np.max(np.negative(A), 0));
    Amax[Amax<0] = 0;
    Amin[Amin<0] = 0;
    r = np.max(Amax * np.ones((n, 1)) + Amin * np.ones((n, 1)));
    u = (.5 / r) * np.ones((n, 1))
    l = -(.5 / r) * np.ones((n, 1))
    outer_t = 1;
    for i in range(1, maxiter):
        # 1. Centering Step: computer x*(outer_t)
        # Just calculate these once
        obj = u - l;
        barrier = b - Amax * u + Amin * l;
        # This is the stupidest way to make diagonal matrices ever.
        # Der(u), then der(l)
        obj_grad = np.array(np.true_divide(1, obj));
        obj_hess = np.array(np.diagflat(
            np.diag(np.true_divide(1, np.square(obj)))));
        barrier_grad = np.array(np.true_divide(1, barrier));
        barrier_hess = np.array(np.diagflat(np.diag(
            np.true_divide(1, np.square(barrier)))));

        # Centering step: This is just Newton's Method
        y = np.dot(outer_t, - np.sum(np.log(obj))) - np.sum(np.log(barrier));
        grad = (outer_t * np.vstack([-obj_grad.T, obj_grad.T]) +
                np.dot(np.vstack([Amax, -Amin]), barrier_grad))
        Atrans = np.hstack([np.reshape(Amax, (2, 1)),
                         np.reshape(-Amin, (2, 1))])
        hess = (outer_t * np.vstack([np.hstack([-obj_hess.T, obj_hess.T]),
                                     np.hstack([obj_hess.T, -obj_hess.T])])
                + np.vstack([Amax, -Amin]).dot(
                    barrier_hess.dot(Atrans)));
        # delX = A\b
        delX = np.linalg.solve(-hess, grad);
        lambdaSq = (np.dot(grad.transpose(),
                           np.dot(np.linalg.pinv(hess), grad)))
        # 2. Stopping Criterion
        # This was a tricky method to find
        if np.any(lambdaSq / 2 < epsilon):
            # We've come to the end of the centering step.
            # If m/t < eps_t, quit. Else, iterate outer_t.
            if 2*m/outer_t < eps_t:
                print u
                print l
                print i
                print outer_t
                return [u, l, outer_t];

            # 4. Increase t
            outer_t = mu * outer_t;

        else:
            du = delX[:,0];
            dl = delX[:,1];
            # Else we keep going!
            # 2. Line Search
            inner_t = 1
            # Have to do some preprocessing here, or else we get NaNs
            # from the log() term
            objStep = (u + du) - (l + dl);
            barrierStep = b - Amax * (u + du) + Amin * (l + dl);
            while np.greater(objStep, 1) | np.greater(barrierStep, 1):
                inner_t = inner_t * beta;
                objStep = (u + inner_t * du) - (l + dl);
                barrierStep = (b - Amax * (u + inner_t * du)
                               + Amin * (l + inner_t *dl));

            yStep = (np.dot(outer_t, - np.sum(np.log(objStep)))
                     - np.sum(np.log(barrierStep)));

            while yStep > (y + alpha * inner_t
                           * np.dot(np.transpose(grad), delX)):
                inner_t = inner_t * beta;
                objStep = (u + inner_t * du) - (l + dl);
                barrierStep = (b - Amax * (u + inner_t * du)
                               + Amin * (l + inner_t *dl));
                yStep = (np.dot(outer_t, - np.sum(np.log(objStep)))
                         - np.sum(np.log(barrierStep)));

            # 3. Update
            l = l + inner_t * dl;
            u = u + inner_t * du;

# MAIN FUNCTION
if __name__ == '__main__':
    m = 5;
    n = 2;
    A = [[0, -1],
         [2, -4],
         [2,  1],
         [-4, 4],
         [-4, 0]];
    b = np.ones((n, 1));
    alpha = 0.2;
    beta = 0.5;
    mu = 20;  # Suggested everywhere in the text.
    maxiter = 2;
    epsilon = .00001;  # For centering method
    eps_t = .000003;  # For t step
    barrierOptimization(A, b, m, n, mu, maxiter, epsilon, eps_t);

