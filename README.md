# gradient-descent
Gradient Descent Algorithm

Implementation of the following algorithms:
    gradient_descent
    gradient_descent_regularization
    cost_function
    cost_function_regularization

Optimization functions:
    find_best_lambda
        Create a list of lambdas.
        Create a set of models with different degrees.
        Iterate through the λ and learn Θ.
        Compute the cross validation error using the learned Θ.
        Return λ and Θ that produces the lowest error on the cross validation set.
    find_best_polynomial_degree  
        Create Polynomial Dataset.
        Optimize Θ using the training set for each polynomial degree.
        Compute the cross validation error using the learned Θ.
        Return polynomial degree and Θ that produces the lowest error on the cross validation set.
