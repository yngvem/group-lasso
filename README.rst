===========
Group Lasso
===========

.. image:: https://pepy.tech/badge/group-lasso
    :target: https://pepy.tech/project/group-lasso
    :alt: PyPI Downloads

.. image:: https://travis-ci.org/yngvem/group-lasso.svg?branch=master
    :target: https://github.com/yngvem/group-lasso

.. image:: https://coveralls.io/repos/github/yngvem/group-lasso/badge.svg
    :target: https://coveralls.io/github/yngvem/group-lasso

.. image:: https://readthedocs.org/projects/group-lasso/badge/?version=latest
    :target: https://group-lasso.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/l/group-lasso.svg
    :target: https://github.com/yngvem/group-lasso/blob/master/LICENSE

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/python/black
    
.. image:: https://www.codefactor.io/repository/github/yngvem/group-lasso/badge
   :target: https://www.codefactor.io/repository/github/yngvem/group-lasso
   :alt: CodeFactor

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

There are several examples that show usage of the library
`here <https://group-lasso.readthedocs.io/en/latest/auto_examples/index.html>`_.

------------
Further work
------------

1. Fully test with sparse arrays and make examples
2. Make it easier to work with categorical data
3. Poisson regression

----------------------
Implementation details
----------------------
The problem is solved using the FISTA optimiser [3]_ with a gradient-based 
adaptive restarting scheme [4]_. No line search is currently implemented, but 
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

.. [3] Beck, A. and Teboulle, M. (2009), A Fast Iterative 
   Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
   SIAM Journal on Imaging Sciences 2009 2:1, 183-202.
   doi:10.1137/080716542  

.. [4] O’Donoghue, B. & Candès, E. (2015), Adaptive Restart for
   Accelerated Gradient Schemes. Found Comput Math 15: 715.
   doi:10.1007/s10208-013-9150-
