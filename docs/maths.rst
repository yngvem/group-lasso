Mathematical background
=======================

Quick overview
--------------

Let us recap the definition of a sparse group lasso regularised machine 
learning algorithm. Consdier the unregularised loss function
:math:`L(\mathbf{\beta}; \mathbf{X}, \mathbf{y})`, where
:math:`\mathbf{\beta}` is the model coefficients, :math:`\mathbf{X}` is the
data matrix and :math:`\mathbf{y}` is the target vector (or matrix in the
case of multiple regression/classification algorithms). Furthermore, we
assume that
:math:`\mathbf{\beta} = \left[\mathbf{\beta}_1^T, ..., \mathbf{\beta}_G^T\right]^T`
and that :math:`\mathbf{X} = \left[\mathbf{X}_1^T, ..., \mathbf{X}_G^T\right]^T`,
where :math:`\mathbf{\beta}_g` and :math:`\mathbf{X}_g` is the coefficients
and data matrices corresponding to covariate group :math:`g`. In this case, we
define the group lasso regularised loss function as

.. math::

    L(\mathbf{\beta}; \mathbf{X}, \mathbf{y})
     + \lambda_1 ||\mathbf{\beta}||_1
     + \lambda_2 \sum_{g \in \mathcal{G}} \sqrt{d_g} ||\mathbf{\beta}||_2

where :math:`\lambda_1` is the parameter-wise regularisation penalty,
:math:`\lambda_2` is the group-wise regularisation penalty,
:math:`\mathbf{\beta}_g \in \mathbf{d_g}` and
:math:`\mathcal{G}` is the set of all groups.

The above regularisation penalty is nice in the sense that it promotes that a
sparse set of groups are chosen for the regularisation coefficients [1]_. 
However, the non-continuous derivative makes the optimisation procedure much
more complicated than with say a Ridge penalty (i.e. squared 2-norm penalty).
One common algorithm used to solve this optimisation problem is 
*group coordinate descent*, in which the optimisation problem is solved for
each group separately, in an alternating fashion. However, I decided to use
the fast iterative soft thresholding (FISTA) algorithm [2]_ with the 
gradient-based restarting scheme given in [3]_. This is regarded as one of the
best algorithms to solve optimisation problems on the form

.. math::

    \text{arg} \min_{\mathbf{\beta}} L(\mathbf{\beta}) + R(\mathbf{\beta}),

where :math:`L` is a convex, differentiable function with Lipschitz continuous
gradient and :math:`R` is a convex lower semicontinouous function. 

Details on FISTA
----------------

There are three essential parts of having an efficient implementation of the
FISTA algorithm. First and foremost, we need an efficient way to compute the
gradient of the loss function. Next, and just as important, we need to be able
to compute the *proximal map* of the regulariser efficiently. That is, we need
to know how to compute

.. math::

    prox(\mathbf{\beta}) = \text{arg} \min_{\hat{\mathbf{\beta}}}
    R(\hat{\mathbf{\beta}}) + \frac{1}{2}||\hat{\mathbf{\beta}} - \mathbf{\beta}||_2^2

efficiently. To compute the proximal map for the sparse group lasso regulariser,
we use the following identity from [4]_:

.. math::

    prox_{\lambda_1 ||\mathbf{\cdot}||_1 + \lambda_2 \sum_g w_g ||\mathbf{\cdot}||}(\mathbf{\beta})
    = prox_{\lambda_2 \sum_g w_g ||\mathbf{\cdot}||}(prox_{\lambda_1 ||\mathbf{\cdot}||_1}(\mathbf{\beta}),

where :math:`prox_{\lambda_1 ||\mathbf{\cdot}||_1 + \lambda_2 \sum_g w_g ||\mathbf{\cdot}||}`
is the proximal map for the sparse group lasso regulariser, 
:math:`prox_{\lambda_2 \sum_g w_g ||\mathbf{\cdot}||}` is the proximal map
for the group lasso regulariser and
:math:`prox_{\lambda_1 ||\mathbf{\cdot}||_1` is the proximal map for the
lasso regulariser. For more information on the proximal map, see [5]_ or [6]_. 
Finally, we need a Lipschitz bound for the gradient of the loss function, since
this is used to compute the step-length of the optimisation procedure. Luckily,
this can also be estimated using a line-search.

Unfortunately, the FISTA algorithm is not stable in the mini-batch case, making
it inefficient for extremely large datasets. However, in my experiments, I have
found that it still recovers the correct sparsity patterns in the data when used
in a mini-batch fashion for the group lasso problem. At least so long as the 
mini-batches are relatively large. 

Computing the Lipschitz coefficients
------------------------------------

The Lipschitz coefficient of the gradient to the sum-of-squares loss is given
by :math:`\sigma_1^2`, where :math:`\sigma_1` is the largest singular value
of the data matrix.

I have not found a published expression for the Lipschitz coefficient of the
sigmoid cross-entropy loss. Therefore, I derived the following bound:

.. math::

    L = \sqrt{12} ||\mathbf{X}||_F,

where :math:`||\mathbf{\cdot}||_F = \sqrt{\sum_{i, j} \mathbf{X}_{i, j}^2}` is
the Frobenius norm. The next step to get group lasso regularised logistic
regression is deriving the Lipschitz bound for the gradient of the softmax
cross-entropy loss.

References
----------
.. [1] Yuan M, Lin Y. Model selection and estimation in regression with
    grouped variables. Journal of the Royal Statistical Society: Series B
    (Statistical Methodology). 2006 Feb;68(1):49-67.
.. [2] Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm
    for linear inverse problems. SIAM journal on imaging sciences.
    2009 Mar 4;2(1):183-202.
.. [3] O’Donoghue B, Candes E. Adaptive restart for accelerated gradient
    schemes. Foundations of computational mathematics.
    2015 Jun 1;15(3):715-32.
.. [4] Yuan L, Liu J, Ye J. (2011), Efficient methods for overlapping group
    lasso. Advances in Neural Information Processing Systems (pp. 352-360).
.. [5] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
    Trends in Optimization, 1(3), 127-239.
.. [6] Beck, A. (2017). First-order methods in optimization (Vol. 25). SIAM.