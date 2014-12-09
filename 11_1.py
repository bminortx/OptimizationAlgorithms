# 11.1
# Grapical representation of the relationship between
# the log barrier function Phi and the minimization
# function f_0

import matplotlib.pyplot as plt
import numpy as np


def phi(x):
    return - np.log(x - 2) - np.log(4 - x)

def f(x):
    return pow(x, 2) + 1

# MAIN FUNCTION
if __name__ == '__main__':
    plt.figure(1)
    x = np.arange(2.01, 3.99, 0.02)
    for t in np.arange(0, 10, .1):
        plt.plot(x, f(x), 'bo', x, f(x)+phi(x)/t, 'k')
        plt.show()
