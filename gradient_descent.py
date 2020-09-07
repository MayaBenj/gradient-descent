import numpy as np
import pandas as pd

"""
Returns Cost Function
"""
def cost_function(m_error, m):
    return (m_error.transpose() @ m_error) / (2 * m)


"""
Returns Cost Function with Regularization
"""
def cost_function_regularization(m_error, m, lam, theta):
    return (m_error.transpose() @ m_error) / (2 * m) + lam / (2*m) * theta.transpose() @ theta


"""
Gradient descent optimization for linear regression
x - numpy array - data matrix (mxn) where m is # of sample and n is # of features
y - numpy array - prediction matrix (mx1)
alpha - float - learning rate
num_iterations - integer
co_criteria - float - convergence criteria

returns learned theta (nx1)
"""
def gradient_descent(x, y, alpha=0.01, num_iterations=100000, co_criteria=0.01):
    m = x.shape[0]  # number of samples
    n = x.shape[1]  # number of features
    prev_cost = 0
    i = 0
    converged = False

    # Initialize theta
    theta = pd.DataFrame(np.random.uniform(-1, 1, size=(n, 1))).values

    while not converged:
        # Hypothesis
        h = x @ theta
        # Mean-error
        m_error = h-y
        # Cost function
        cost = cost_function(m_error, m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # Calculate derivative of cost function
        gradient = 1 / m * (x.transpose() @ m_error)
        # Update theta
        theta = theta - alpha * gradient
        if (abs(prev_cost - cost) < co_criteria) or i == num_iterations:
            converged = True
        else:
            prev_cost = cost
            i = i + 1
    return theta


"""
Gradient descent optimization for liner regression with regularization
x - data matrix (mxn) where m is sample and n is features
y - prediction matrix (mx1)
lam - regularization parameter
alpha - learning rate
num_iterations 
co_criteria - convergence criteria

returns learned theta (nx1)
"""
def gradient_descent_regularization(x, y, lam=0.01, alpha=0.01, num_iterations=100000, co_criteria=0.01):
    m = x.shape[0]  # number of samples
    n = x.shape[1]  # number of features
    prev_cost = 0
    i = 0
    converged = False

    # Initialize theta
    theta = pd.DataFrame(np.random.uniform(-1, 1, size=(n, 1))).values

    while not converged:
        # Hypothesis
        h = x @ theta
        # Mean-error
        m_error = h-y
        # Cost function
        cost = cost_function_regularization(m_error, m, lam, theta)
        print("Iteration %d | Cost: %f" % (i, cost))
        # Calculate derivative of cost function
        gradient = 1 / m * (x.transpose() @ m_error)
        # Update theta, do not regularize theta0
        theta[0, :] = theta[0, :] - alpha * gradient[0, :]
        theta[1:, :] = theta[1:, :] * (1 - alpha * lam / m) - alpha * gradient[1:, :]
        if (abs(prev_cost - cost) < co_criteria) or i == num_iterations:
            converged = True
        else:
            prev_cost = cost
            i = i + 1
    return theta
