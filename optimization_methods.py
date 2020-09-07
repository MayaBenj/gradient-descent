from gradient_descent import gradient_descent_regularization, cost_function
import numpy as np

"""
Get list of numbers where each entry is the previous entry * 2 
starting with 0 and initial_value, for iterations
"""
def get_lambdas(initial_value=0.01, iter=11):
    list = [0, initial_value]
    for i in range(1, iter):
        list.append(list[i] * 2)
    return list


"""
Given train and cross validation data find lambda regularization parameter that best fits data
Also returns the corresponding theta
"""
def find_best_lambda(x_train, y_train, x_cv, y_cv):
    lambda_list = get_lambdas()
    min_theta = []
    for lam in lambda_list:
        min_theta.append(gradient_descent_regularization(x_train, y_train, lam=lam, alpha=0.2, co_criteria=0.0001))

    J_cv = []
    m_cv = x_cv.shape[0]
    for theta in min_theta:
        J_cv.append(cost_function(x_cv @ theta - y_cv, m_cv)[0, 0])
    return lambda_list[min(J_cv)], min_theta[min(J_cv)]


"""
Get data-set x and return a list of data-set to the d polynomial degree
List index matches degree, 0 is set to x0 vector.

x - numpy array
d - integer
"""
def create_polynomial_dataset(x, d):
    dataset_list = [np.ones(x.shape[0])]
    for i in range(1, d+1):
        dataset_list.append(np.column_stack((dataset_list[i - 1], x[:, 1:] ** i)))
    return dataset_list


"""
Given train and cross validation data find polynomial degree that fits data best from 1 to d
Also returns the corresponding theta
"""
def find_best_polynomial_degree(d, x_train, y_train, x_cv, y_cv):
    x_train_list = create_polynomial_dataset(x_train, d)

    # Optimize theta for each hypothesis
    min_theta = [0]  # initializing 0 do index will start from 1
    for x_train in x_train_list[1:]:
        min_theta.append(gradient_descent(x_train, y_train))

    # Calculate cv error for each polynomial degree
    x_cv_list = create_polynomial_dataset(x_cv, d)
    m_cv = x_cv.shape[0]
    J_cv = [0]  # initializing 0 do index will start from 1
    for i in range(1, len(x_cv_list)):
        J_cv.append(cost_function(x_cv_list[i] @ min_theta[i] - y_cv, m_cv)[0, 0])

    # Find polynomial degree with least error
    d = J_cv.index(min(J_cv[1:]))

    return d, min_theta[d]
