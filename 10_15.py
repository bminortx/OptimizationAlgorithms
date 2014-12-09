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

        yStep = (x + t * delX).T * np.log(x + t * delX);
        while np.all(yStep > y + alpha * t * lambdaSq):
            t = t * beta;
            yStep = (x + t * delX).T * np.log(x + t * delX);

        # 3. Update to next step
        x = x + t * delX;


# Remember: the backtracking search on the Infeasible Newton Method is
# on the residual dual and primal, not the term itself.
def solveInfeasibleNewton(A, x, b, n, p, alpha, beta, maxiter, epsilon):
    # Provide the dual variable we're solving for
    nu = np.zeros((p, 1));
    for i in range(1, maxiter):
        # Notice: we don't calculate a term for y.
        grad = 1 + np.log(x);
        # Use the diagonal Hessian trick!
        hess = np.diagflat(np.true_divide(1, x))
        # The Infeasible solution is also a KKT
        # bsol here also represents
        # [ r_dual ]
        # [r_primal]
        # Which we will use during the backtracking search
        Asol = np.array(np.vstack([np.hstack([hess, A.T]),
                                   np.hstack([A, np.zeros((p, p))])]));
        residSol = np.array(np.vstack([grad + np.dot(A.T, nu),
                                       np.dot(A, x) - b]));
        # Since the solution above is stacked, we only need the
        # first n elements as the sollution to x.
        sol = np.linalg.solve(-Asol, residSol);
        delX = sol[0:n];
        delNu = sol[n:];
        # Cutoff point, checked after step 1
        if np.any(np.linalg.norm(residSol) <= epsilon):
            # print "residual: "
            # print residSol
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

        gradStep = 1 + np.log(x + t * delX);
        residStep = np.array(np.vstack([gradStep
                                        + np.dot(A.T, (nu + t * delNu)),
                                        np.dot(A, (x + t * delX)) - b]));
        while np.all(residStep > (1 - alpha * t) * np.linalg.norm(residSol)):
            t = t * beta;
            gradStep = 1 + np.log(x + t * delX);
            residStep = np.array(np.vstack([gradStep
                                            + np.dot(A.T * (nu + t * delNu)),
                                            np.dot(A, (x + t * delNu)) - b]));

        # 3. Update to next step
        x = x + t * delX;
        nu = nu + t * delNu;


# Literally the same as the primal Newton, just using the
# dual equation instead.
def solveDualNewton(A, x, b, n, p, alpha, beta, maxiter, epsilon):
    # Provide the dual variable we're solving for
    nu = np.zeros((p, 1));
    for i in range(1, maxiter):
        # This requires that we find the dual formula, which here is:
        # maximize -b' * nu - sum( exp( -a' * nu - 1))
        # So take the negative and minimize!
        y = np.dot(b.T, nu) + np.sum(np.exp(np.dot(-A.T, nu) - 1));
        # Thus, the grad and hess are derived from this expression
        # w.r.t. nu
        grad = b - np.dot(A, np.exp(np.dot(-A.T, nu) - 1));
        # Use the diagonal Hessian trick! Because it's a sum.
        hess = np.dot(
            np.dot(A,
                   np.diagflat(np.exp(np.dot(-A.T, nu) - 1))),
            A.T);
        # This is just a straightforward linear system
        delNu = np.linalg.solve(-hess, grad);
        lambdaSq = np.dot(grad.T, delNu);
        # Cutoff point, checked after step 1
        if np.any(np.abs(lambdaSq) <= epsilon):
            print "Iterations";
            print i;
            return x;

        # Else we keep going!
        # 2. Line Search
        t = 1;
        yStep = (np.dot(-b.T, nu + t * delNu)
                 - np.sum(np.exp(np.dot(-A.T, (nu + t * delNu)) - 1)));
        while np.all(yStep > y + alpha * t * lambdaSq):
            t = t * beta;
            yStep = (np.dot(-b.T, nu + t * delNu)
                     - np.sum(np.exp(np.dot(-A.T, (nu + t * delNu)) - 1)));

        # 3. Update to next step
        nu = nu + t * delNu;


# MAIN FUNCTION
if __name__ == '__main__':
    alpha = 0.01;
    beta = 0.5;
    maxiter = 100;
    epsilon = 1e-10;
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
    solveInfeasibleNewton(A, x, b, n, p, alpha, beta, maxiter, epsilon);
    solveDualNewton(A, x, b, n, p, alpha, beta, maxiter, epsilon);
