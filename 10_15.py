# Function: minimize
# f(x) = sum( x * log(x) )
# subject to Ax = b
# We use three methods to solve:
# 1. Newton's Method
# 2. Infeasible Newton's Method
# 3. Dual Newton's Method

import matplotlib.pyplot as plt
import numpy as np


# WORKS
def solveNewton(A, x, b, n, p, alpha, beta, maxiter, epsilon):
    for i in range(1, maxiter):
        y = x.T * np.log(x)
        grad = 1 + np.log(x)
        # Use the diagonal Hessian trick!
        hess = np.diagflat(np.true_divide(1, x))
        # Since this has equality constraints, we have to use
        # the KKT Solution
        Asol = np.array(np.vstack([np.hstack([hess, A.T]),
                                   np.hstack([A, np.zeros((p, p))])]));
        bsol = np.array(np.vstack([grad, np.zeros((p, 1))]));
        # Since the solution above is stacked, we only need the
        # first n elements as the sollution to x. 
        sol = np.linalg.solve(-Asol, bsol);
        delX = sol[0:n];
        lambdaSq = np.dot(grad.T, delX);
        # Cutoff point, checked after step 1
        if np.any(np.abs(lambdaSq) <= epsilon):
            print "lambdaSq"
            print lambdaSq
            print "Iterations";
            print i;
            return x;

        # Else we keep going!
        # 2. Line Search
        t = 1;
        # Have to do some preprocessing here, or else we get NaNs
        # from the log() term
        while np.all(x + t * delX < 0):
            t = t * beta;
            print t

        yStep = (x + t * delX).T * np.log(x + t * delX);
        while np.all(yStep > y + alpha * t * lambdaSq):
            t = t * beta;
            yStep = (x + t * delX).T * np.log(x + t * delX);

        # 3. Update to next step
        x = x + t * delX;


def solveInfeasibleNewton(A, x, b, alpha, beta, maxiter, epsilon):
    for i in range(1, maxiter):
        y = x.T * np.log(x)
        grad = 1 + np.log(x)
        # Use the diagonal Hessian trick!
        hess = np.diagflat(np.true_divide(1, x))
        # delX = A\b
        delX = np.linalg.solve(-hess, grad);
        lambdaSq = np.dot(grad.T, delX);
        # Cutoff point, checked after step 1
        if np.any(np.abs(lambdaSq) <= epsilon):
            print "lambdaSq"
            print lambdaSq
            print "Iterations";
            print i;
            return x;

        # Else we keep going!
        # 2. Line Searchn
        t = 1;
        # Have to do some preprocessing here, or else we get NaNs
        # from the log() term
        while np.all(x + t * delX < 0):
            t = t * beta;
            print t

        yStep = (x + t * delX).T * np.log(x + t * delX);
        while np.all(yStep > y + alpha * t * lambdaSq):
            t = t * beta;
            yStep = (x + t * delX).T * np.log(x + t * delX);

        # 3. Update to next step
        x = x + t * delX;


def solveDualNewton(A, x, b, alpha, beta, maxiter, epsilon):
    T = np.array([])
    for i in range(1, maxiter):
        y = (- np.sum(np.log(1 - np.dot(A, x)))
             - np.sum(np.log(1 + x))
             - np.sum(np.log(1 - x)));
        grad = (np.dot(np.transpose(A), np.true_divide(1, (1 - np.dot(A, x))))
                - np.true_divide(1, 1 + x)
                + np.true_divide(1, 1 - x));
        # JESUS That was hard to organize
        hess = (np.dot(A.transpose(),
                       np.dot(
                           np.diagflat(
                               np.square(np.true_divide(1, 1 - np.dot(A, x)))),
                           A))
                + np.diagflat(np.true_divide(1, np.square(1 + x)))
                + np.diagflat(np.true_divide(1, np.square(1 - x))))
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
    alpha = 0.01;
    beta = 0.5;
    maxiter = 100;
    epsilon = 1e-5;
    n = 100;
    p = 30;
    # CREATE OUR VARIABLES
    # A: p x n. Created from standard distribution
    A = np.random.rand(p, n);
    # x: n x 1
    x = np.random.rand(n, 1);
    # b is derived from Ax = b, making x feasible
    b = np.dot(A, x);
    solveNewton(A, x, b, n, p, alpha, beta, maxiter, epsilon);
    # solveInfeasibleNewton(A, x, b, alpha, beta, maxiter, epsilon);
    # solveDualNewton(A, x, b, alpha, beta, maxiter, epsilon);
