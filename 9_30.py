# 9.30
# Plot the objective function and step length versus iteration
# number. (Once you have determined p to high accuracy, you can also
# plot f - p versus iteration.) Experiment with the backtracking
# parameters alpha and beta  to see their effect on the total number of
# iterations required. Carry these experiments out for several instances
# of the problem of different sizes.

import matplotlib.pyplot as plt
import scipy
import numpy as np

# Gradient method
# WORKS
def solve_grad(A, x, alpha, beta, maxiter, epsilon):
    for i in range(1, maxiter):
        y = (- np.sum(np.log(1 - np.dot(A, x)))
             - np.sum(np.log(1 + x))
             - np.sum(np.log(1 - x)));
        grad = (np.dot(np.transpose(A), np.divide(1, (1 - np.dot(A, x))))
                - np.divide(1, 1 + x)
                + np.divide(1, 1 - x));
        delX = -grad
        # Cutoff point, checked after step 1
        if np.linalg.norm(grad) <= epsilon:
            print "Iterations"
            print i;
            return x;

        # Else we keep going!
        # 2. Line Search
        t = 1
        # Have to do some preprocessing here, or else we get NaNs
        # from the log() term
        dotChange = np.max(np.dot(A, (x + t * delX)));
        squareChange = np.max(np.abs(x + t * delX));
        while np.greater(dotChange, 1) | np.greater(squareChange, 1):
            t = t * beta;
            dotChange = np.max(np.dot(A, (x + t * delX)));
            squareChange = np.max(np.abs(x + t * delX));

        yStep = (- np.sum(np.log(1 - np.dot(A, (x + t * delX))))
                 - np.sum(np.log(1 - np.square(x + t * delX))))
        while yStep > y + alpha * t * np.dot(np.transpose(grad), delX):
            t = t * beta;
            yStep = (- np.sum(np.log(1 - np.dot(A, (x + t * delX))))
                     - np.sum(np.log(1 - np.square(x + t * delX))))

        # 3. Update to next step
        x = x + t * delX;


def solve_newton(A, x, alpha, beta, maxiter, epsilon):
    T = np.array([])
    for i in range(1, maxiter):
        y = (- np.sum(np.log(1 - np.dot(A, x)))
             - np.sum(np.log(1 + x))
             - np.sum(np.log(1 - x)));
        grad = (np.dot(np.transpose(A), np.divide(1, (1 - np.dot(A, x))))
                - np.divide(1, 1 + x)
                + np.divide(1, 1 - x));
        # JESUS That was hard to organize
        hess = (np.dot(A.transpose(),
                       np.dot(
                           np.diagflat(
                               np.square(np.divide(1, 1 - np.dot(A, x)))),
                           A))
                + np.diagflat(np.divide(1, np.square(1 + x)))
                + np.diagflat(np.divide(1, np.square(1 - x))))
        # delX = A\b
        delX = np.linalg.solve(-hess, grad)
        lambdaSq = (np.dot(grad.transpose(),
                           np.dot(np.linalg.pinv(hess), grad)))
        # Cutoff point, checked after step 1
        if lambdaSq / 2 <= epsilon:
            print "Iterations"
            print i;
            # Graph this later
            plt.scatter(T);
            return x;

        # Else we keep going!
        # 2. Line Search
        t = 1
        # Have to do some preprocessing here, or else we get NaNs
        # from the log() term
        dotChange = np.max(np.dot(A, (x + t * delX)));
        squareChange = np.max(np.abs(x + t * delX));
        while np.greater(dotChange, 1) | np.greater(squareChange, 1):
            t = t * beta;
            dotChange = np.max(np.dot(A, (x + t * delX)));
            squareChange = np.max(np.abs(x + t * delX));

        yStep = (- np.sum(np.log(1 - np.dot(A, (x + t * delX))))
                 - np.sum(np.log(1 - np.square(x + t * delX))))
        while yStep > y + alpha * t * np.dot(np.transpose(grad), delX):
            t = t * beta;
            yStep = (- np.sum(np.log(1 - np.dot(A, (x + t * delX))))
                     - np.sum(np.log(1 - np.square(x + t * delX))))

        # 3. Update to next step
        np.append(T, t);
        x = x + t * delX;


# MAIN FUNCTION
if __name__ == '__main__':
    alpha = 0.5;
    beta = 0.2;
    maxiter = 100;
    epsilon = 1e-5;
    n = 5;
    m = 8;
    # CREATE OUR VARIABLES
    # A: m x n. Created from standard distribution
    A = np.random.randn(m, n)
    # x: n x 1
    x = np.zeros((n, 1))
    solve_grad(A, x, alpha, beta, maxiter, epsilon);
    solve_newton(A, x, alpha, beta, maxiter, epsilon);
