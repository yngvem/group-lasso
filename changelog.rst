Changelog
=========

Version 1.5.0
-------------
 * Improved scikit-learn compatibility
 * Fixed bug where fitted estimators were unpicklable

Version 1.4.1
-------------
 * Fixed so GroupLasso objects can be initialised without specifying the groups
 * Fixed bug with viewing the selected groups
 * Changed to ConvergenceWarning instead of RuntimeWarning for when FISTA doesn't converge
 * Fixed typo in warning message
 * Fixed bug with list input for the groups
 * Added matplotlib to requirements.txt
 * Changed how groups are parsed, should hopefully fix some unreported bugs

Version 1.4.0
-------------

 * Fixed bug with how LogisticGroupLasso applied intercept in some methods
 * Implemented backtracking line search for Lipschitz
 * Added centering-based preconditioning for dense design matrices

Version 1.2.0
-------------

 * Merged logistic group lasso and multinomial group lasso to one class.
 * Added support for sparse matrices.
 * Added utility for extracting groups from a sklearn OneHotEncoder.
 * Added option to control group-wise scaling of regularisation.

Version 1.2.2
-------------

 * Fixed bug where the old_regularisation and supress_warning arguments could only
   be set by the base group lasso class, but none of the actual models.

Version 1.2.1
-------------

 * Fixed bug where the regularisation was multiplied with the Lipschitz coefficient
   of the loss gradient.

Version 1.1.1
-------------

 * Added warm restarts

Version 1.0.0
-------------

 * Started changelog
 * Working group LASSO regularised least squares, logistic and multinomial regression.

