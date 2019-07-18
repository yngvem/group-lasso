Efficient Group Lasso in Python
===============================

What is group lasso?
--------------------

It is often the case that we have a dataset where the covariates form natural
groups. These groups can represent biological function in gene expression
data or maybe sensor location in climate data. We then wish to find a sparse
subset of these covariate groups that describe the relationship in the data.
Let us look at an example to crystalise the usefulness of this further.

Say that we work as data scientists for a large Norwegian food supplier and
wish to make a prediction model for the amount of that will be sold based on
weather data. We have weather data from cities in Norway and need to know how
the fruit should be distributed across different warehouses. From each city,
we have information about temperature, precipitation, wind strength, wind
direction and how cloudy it is. Multiplying the number of cities with the
number of covariates per city, we get 1500 different covariates in total.
It is unlikely that we need all these covariates in our model, so we seek a 
sparse set of these to do our predictions with.

Let us now assume that the weather data API that we use charge money by
the number of cities we query, but the amount of information we get per
city. We therefore wish to create a regression model that predicts fruit
demand based on a sparse set of city observations. One way to achieve such
sparsity is through the framework of group lasso regularisation [1]_.


A quick mathematical interlude
------------------------------

Let us now briefly describe the mathematical problem solved in group lasso
regularised machine learning problems. Originally, group lasso algorithm [1]_
was defined as regularised linear regression with the following loss function

.. math::

    \text{arg} \min_{\mathbf{\beta}_g \in \mathbb{R^{d_g}}} 
    || \sum_{g \in \mathcal{G}} \left[\mathbf{X}_g\mathbf{\beta}_g\right] - \mathbf{y} ||_2^2
    + w \sum_{g \in \mathcal{G}} \sqrt{d_g}||\mathbf{\beta}_g||_2,

where :math:`\mathbf{X}_g \in \mathbb{R}^{n \times d_g}` is the data matrix
corresponding to the covariates in group :math:`g`, :math:`\mathbf{\beta}_g`
is the regression coefficients corresponding to group :math:`g`, 
:math:`\mathbf{y} \in \mathbf{R}^n` is the regression target, :math:`n` is the
number of measurements, :math:`d_g` is the dimensionality of group :math:`g`,
:math:`w` is the regularisation penalty and :math:`\mathcal{G}` is the set of
all groups. 

Notice, in the equation above, that the 2-norm is *not* squared. A consequence
of this is that the regulariser has a "kink" at zero, uninformative covariate
groups to have zero-valued regression coefficients. Later, it has been popular
to use this methodology to regularise other machine learning algorithms, such
as logistic regression. The "only" thing neccesary to do this is to exchange
the squared norm term, :math:`|| \sum_{g \in \mathcal{G}} \left[\mathbf{X}_g\mathbf{\beta}_g\right] - \mathbf{y} ||_2^2`,
with a general loss term, :math:`L(\mathbf{\beta}; \mathbf{X}, \mathbf{y})`,
where :math:`\mathbf{\beta}` and :math:`\mathbf{X}` is the concatenation
of all group coefficients and group data matrices, respectively.


API design
----------

The ``group-lasso`` python library is modelled after the ``scikit-learn`` API
and should be fully compliant with the ``scikit-learn`` ecosystem.
Consequently, the ``group-lasso`` library depends on ``numpy``, ``scipy``
and ``scikit-learn``.

Currently, the only supported algorithm is group-lasso regularised linear
and multiple regression, which is available in the ``group_lasso.GroupLasso``
class. However, I am working on an experimental class with group lasso
regularised logistic regression, which is available in the 
``group_lasso.LogisticGroupLasso`` class. Currently, this class only supports
binary classification problems through a sigmoidal transformation, but
I am working on a multiple classification algorithm with the softmax
transformation.

All classes in this library is implemented as both ``scikit-learn``
transformers and their regressors or classifiers (dependent on their
use case). The reason for this is that to use lasso based models for
variable selection, the regularisation coefficient should be quite high,
resulting in sub-par performance on the actual task of interest. Therefore,
it is common to first use a lasso-like algorithm to select the relevant
features before using another another algorithm (say ridge regression)
for the task at hand. Therefore, the ``transform`` method of
``group_lasso.GroupLasso`` to remove the columns of the input dataset
corresponding to zero-valued coefficients.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   examples
   maths
   api_reference
   

References
----------
.. [1] Yuan M, Lin Y. Model selection and estimation in regression with
    grouped variables. Journal of the Royal Statistical Society: Series B
    (Statistical Methodology). 2006 Feb;68(1):49-67.