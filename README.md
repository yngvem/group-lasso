# Group Lasso
The group lasso [1] regulariser is a well known method to achieve structured sparsity
in machine learning and statistics. The idea is to create non-overlapping groups of
covariate, and recover regression weights in which only a sparse set of these covariate
groups have non-zero components.

There are several reasons for why this might be a good idea. Say for example that we have
a set of sensors and each of these sensors generate five measurements. We don't want 
to maintain an unneccesary number of sensors. If we try normal LASSO regression, then
we will get sparse components. However, these sparse components might not correspond
to a sparse set of sensors, since they each generate five measurements. If we instead
use group LASSO with measurements grouped by which sensor they were measured by, then
we will get a sparse set of sensors.

# About this project
This project is developed by Yngve Mardal Moe and released under an MIT lisence.

## Todos:
The todos are, in decreasing order of importance

 1. Write a better readme
    - Code examples
    - Installation guide (after point 2.)
    - Better description of Group LASSO
 2. Make easy to install
    - Create setup.py
    - Create wheels
    - Upload to PyPI
 3. Write more docstrings
 4. Python 3.5 compatibility
 5. Better ScikitLearn compatibility
    - Use Mixins?
    - Use randomness correctly
 6. Multiple regression
 7. Classification problems (I have an experimental implementation, but it's not tested yet)

Unfortunately, the most interesting parts are the least important ones, so expect the list
to be worked on from both ends simultaneously.

## Implementation details
The problem is solved using the FISTA optimiser [2] with a gradient-based adaptive restarting scheme [3]. No line search is currently implemented, but I hope to look at that later.

Although fast, the FISTA optimiser does not achieve as low loss values as the significantly slower second order interior point methods. This might, at first glance, seem like a problem. However, it does recover the sparsity patterns of the data, which can be used to train a new model with the given subset of the features.

Also, even though the FISTA optimiser is not meant for stochastic optimisation, it has to my experience not suffered a large fall in performance when the mini batch was large enough. I have therefore implemented mini-batch optimisation using FISTA, and thus been able to fit models based on data with ~500 columns and 10 000 000 rows on my moderately priced laptop.

Finally, we note that since FISTA uses Nesterov acceleration, is not a descent algorithm. We can therefore not expect the loss to decrease monotonically.

## References
[1]: Yuan, M. and Lin, Y. (2006), Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68: 49-67. doi:10.1111/j.1467-9868.2005.00532.x

[2]: Beck, A. and Teboulle, M. (2009), A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences 2009 2:1, 183-202. doi:10.1137/080716542  
[3]: O’Donoghue, B. & Candès, E. (2015), Adaptive Restart for Accelerated Gradient Schemes. Found Comput Math 15: 715. doi:10.1007/s10208-013-9150-
