===========
Group Lasso
===========

.. image:: https://coveralls.io/repos/github/yngvem/group-lasso/badge.svg
    :target: https://coveralls.io/github/yngvem/group-lasso

.. image:: https://travis-ci.org/yngvem/group-lasso.svg?branch=master
    :target: https://github.com/yngvem/group-lasso

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/python/black

.. image:: https://img.shields.io/pypi/l/group-lasso.svg
    :target: https://github.com/yngvem/group-lasso/blob/master/LICENSE

.. image:: https://readthedocs.org/projects/group-lasso/badge/?version=latest
    :target: https://group-lasso.readthedocs.io/en/latest/?badge=latest

The group lasso [1]_ regulariser is a well known method to achieve structured 
sparsity in machine learning and statistics. The idea is to create 
non-overlapping groups of covariates, and recover regression weights in which 
only a sparse set of these covariate groups have non-zero components.

There are several reasons for why this might be a good idea. Say for example 
that we have a set of sensors and each of these sensors generate five 
measurements. We don't want to maintain an unneccesary number of sensors. 
If we try normal LASSO regression, then we will get sparse components. 
However, these sparse components might not correspond to a sparse set of 
sensors, since they each generate five measurements. If we instead use group 
LASSO with measurements grouped by which sensor they were measured by, then
we will get a sparse set of sensors.

An extension of the group lasso regulariser is the sparse group lasso
regulariser [2]_, which imposes both group-wise sparsity and coefficient-wise
sparsity. This is done by combining the group lasso penalty with the
traditional lasso penalty. In this library, I have implemented an efficient
sparse group lasso solver being fully scikit-learn API compliant.

------------------
About this project
------------------
This project is developed by Yngve Mardal Moe and released under an MIT 
lisence. I am still working out a few things so changes might come rapidly.

------------------
Installation guide
------------------
Group-lasso requires Python 3.5+, numpy and scikit-learn. 
To install group-lasso via ``pip``, simply run the command::

    pip install group-lasso

Alternatively, you can manually pull this repository and run the
``setup.py`` file::

    git clone https://github.com/yngvem/group-lasso.git
    cd group-lasso
    python setup.py

-------------
Documentation
-------------

You can read the full documentation on 
`readthedocs <https://group-lasso.readthedocs.io/en/latest/maths.html>`_.

--------
Examples
--------

Group lasso regression
======================

The group lasso regulariser is implemented following the scikit-learn API,
making it easy to use for those familiar with the Python ML ecosystem.

.. code-block:: python

    import numpy as np
    from group_lasso import GroupLasso

    # Dataset parameters
    num_data_points = 10_000
    num_features = 500
    num_groups = 25
    assert num_features % num_groups == 0

    # Generate data matrix
    X = np.random.standard_normal((num_data_points, num_features))

    # Generate coefficients and intercept
    w = np.random.standard_normal((500, 1))
    intercept = 2

    # Generate groups and randomly set coefficients to zero
    groups = np.array([[group]*20 for group in range(25)]).ravel()
    for group in range(num_groups):
        w[groups == group] *= np.random.random() < 0.8
    
    # Generate target vector:
    y = X@w + intercept
    noise = np.random.standard_normal(y.shape)
    noise /= np.linalg.norm(noise)
    noise *= 0.3*np.linalg.norm(y)
    y += noise

    # Generate group lasso object and fit the model
    gl = GroupLasso(groups=groups, reg=.05)
    gl.fit(X, y)
    estimated_w = gl.coef_
    estimated_intercept = gl.intercept_[0]

    # Evaluate the model
    coef_correlation = np.corrcoef(w.ravel(), estimated_w.ravel())[0, 1]
    print(
        "True intercept: {intercept:.2f}. Estimated intercept: {estimated_intercept:.2f}".format(
            intercept=intercept,
            estimated_intercept=estimated_intercept
         )
    )
    print(
        "Correlation between true and estimated coefficients: {coef_correlation:.2f}".format(
            coef_correlation=coef_correlation
         )
    )
    
.. code-block::

    True intercept: 2.00. Estimated intercept: 1.53
    Correlation between true and estimated coefficients: 0.98


Group lasso as a transformer
============================

Group lasso regression can also be used as a transformer

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from group_lasso import GroupLasso

    # Dataset parameters
    num_data_points = 10_000
    num_features = 500
    num_groups = 25
    assert num_features % num_groups == 0

    # Generate data matrix
    X = np.random.standard_normal((num_data_points, num_features))

    # Generate coefficients and intercept
    w = np.random.standard_normal((500, 1))
    intercept = 2

    # Generate groups and randomly set coefficients to zero
    groups = np.array([[group]*20 for group in range(25)]).ravel()
    for group in range(num_groups):
        w[groups == group] *= np.random.random() < 0.8
    
    # Generate target vector:
    y = X@w + intercept
    noise = np.random.standard_normal(y.shape)
    noise /= np.linalg.norm(noise)
    noise *= 0.3*np.linalg.norm(y)
    y += noise

    # Generate group lasso object and fit the model
    # We use an artificially high regularisation coefficient since
    #  we want to use group lasso as a variable selection algorithm.
    gl = GroupLasso(groups=groups, group_reg=0.1, l1_reg=0.05)
    gl.fit(X, y)
    new_X = gl.transform(X)


    # Evaluate the model
    predicted_y = gl.predict(X)
    R_squared = 1 - np.sum((y - predicted_y)**2)/np.sum(y**2)

    print("The rows with zero-valued coefficients have now been removed from the dataset.")
    print("The new shape is:", new_X.shape)
    print("The R^2 statistic for the group lasso model is: {R_squared:.2f}".format(R_squared=R_squared))
    print("This is very low since the regularisation is so high."

    # Use group lasso in a scikit-learn pipeline
    pipe = Pipeline(
        memory=None,
        steps=[
            ('variable_selection', GroupLasso(groups=groups, reg=.1)),
            ('regressor', Ridge(alpha=0.1))
        ]
    )
    pipe.fit(X, y)
    predicted_y = pipe.predict(X)
    R_squared = 1 - np.sum((y - predicted_y)**2)/np.sum(y**2)

    print("The R^2 statistic for the pipeline is: {R_squared:.2f}".format(R_squared=R_squared))

    
.. code-block::

    The rows with zero-valued coefficients have now been removed from the dataset.
    The new shape is: (10000, 280)
    The R^2 statistic for the group lasso model is: 0.17
    This is very low since the regularisation is so high.
    The R^2 statistic for the pipeline is: 0.72

-----------
Furher work
-----------
The todos are, in decreasing order of importance

1. Python 3.5 compatibility

----------------------
Implementation details
----------------------
The problem is solved using the FISTA optimiser [4]_ with a gradient-based 
adaptive restarting scheme [5]_. No line search is currently implemented, but 
I hope to look at that later.

Although fast, the FISTA optimiser does not achieve as low loss values as the 
significantly slower second order interior point methods. This might, at 
first glance, seem like a problem. However, it does recover the sparsity 
patterns of the data, which can be used to train a new model with the given 
subset of the features.

Also, even though the FISTA optimiser is not meant for stochastic 
optimisation, it has to my experience not suffered a large fall in 
performance when the mini batch was large enough. I have therefore 
implemented mini-batch optimisation using FISTA, and thus been able to fit 
models based on data with ~500 columns and 10 000 000 rows on my moderately 
priced laptop.

Finally, we note that since FISTA uses Nesterov acceleration, is not a 
descent algorithm. We can therefore not expect the loss to decrease 
monotonically.

----------
References
----------

.. [1] Yuan, M. and Lin, Y. (2006), Model selection and estimation in
   regression with grouped variables. Journal of the Royal Statistical
   Society: Series B (Statistical Methodology), 68: 49-67.
   doi:10.1111/j.1467-9868.2005.00532.x

.. [2] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013).
    A sparse-group lasso. Journal of Computational and Graphical
    Statistics, 22(2), 231-245.

.. [3] Yuan L, Liu J, Ye J. (2011), Efficient methods for overlapping
   group lasso. Advances in Neural Information Processing Systems
   (pp. 352-360).

.. [4] Beck, A. and Teboulle, M. (2009), A Fast Iterative 
   Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
   SIAM Journal on Imaging Sciences 2009 2:1, 183-202.
   doi:10.1137/080716542  

.. [5] O’Donoghue, B. & Candès, E. (2015), Adaptive Restart for
   Accelerated Gradient Schemes. Found Comput Math 15: 715.
   doi:10.1007/s10208-013-9150-
