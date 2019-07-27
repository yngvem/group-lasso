Examples
========

Group lasso regression
----------------------

The group lasso regulariser is implemented following the scikit-learn API,
making it easy to use for those familiar with the Python ML ecosystem.

.. code-block:: python

    import numpy as np
    from group_lasso import GroupLasso

    # Dataset parameters
    num_data_points = 10000
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
    gl = GroupLasso(groups=groups, group_reg=0.05, l1_reg=0.01)
    gl.fit(X, y)
    estimated_w = gl.coef_
    estimated_intercept = gl.intercept_[0]

    # Evaluate the model
    coef_correlation = np.corrcoef(w.ravel(), estimated_w.ravel())[0, 1]
    print(
        "True intercept: {intercept:.2f}. Estimated intercept: {estimated_intercept:.2f}".format(
            estimated_intercept=estimated_intercept
        )
    )
    print(
        "Correlation between true and estimated coefficients: {coef_correlation:.2f}".format(
            coef_correlation=coef_correlation
         )
    )

.. code-block:: none

    True intercept: 2.00. Estimated intercept: 1.53
    Correlation between true and estimated coefficients: 0.98


Group lasso as a transformer
----------------------------

Group lasso regression can also be used as a transformer

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from group_lasso import GroupLasso

    # Dataset parameters
    num_data_points = 10000
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

    
.. code-block:: none

    The rows with zero-valued coefficients have now been removed from the dataset.
    The new shape is: (10000, 280)
    The R^2 statistic for the group lasso model is: 0.17
    This is very low since the regularisation is so high.
    The R^2 statistic for the pipeline is: 0.72
