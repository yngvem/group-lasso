"""
GroupLasso as a transformer
============================

A sample script to demonstrate how the group lasso estimators can be used
for variable selection in a scikit-learn pipeline.
"""

###############################################################################
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from group_lasso import GroupLasso

np.random.seed(0)


###############################################################################
# Set dataset parameters
# ----------------------
group_sizes = [np.random.randint(10, 20) for i in range(50)]
active_groups = [np.random.randint(2) for _ in group_sizes]
groups = np.concatenate(
    [size * [i] for i, size in enumerate(group_sizes)]
).reshape(-1, 1)
num_coeffs = sum(group_sizes)
num_datapoints = 10000
noise_std = 20


###############################################################################
# Generate data matrix
# --------------------
X = np.random.standard_normal((num_datapoints, num_coeffs))


###############################################################################
# Generate coefficients
# ---------------------
w = np.concatenate(
    [
        np.random.standard_normal(group_size) * is_active
        for group_size, is_active in zip(group_sizes, active_groups)
    ]
)
w = w.reshape(-1, 1)
true_coefficient_mask = w != 0
intercept = 2


###############################################################################
# Generate regression targets
# ---------------------------
y_true = X @ w + intercept
y = y_true + np.random.randn(*y_true.shape) * noise_std


###############################################################################
# View noisy data and compute maximum R^2
# ---------------------------------------
plt.figure()
plt.plot(y, y_true, ".")
plt.xlabel("Noisy targets")
plt.ylabel("Noise-free targets")
# Use noisy y as true because that is what we would have access
# to in a real-life setting.
R2_best = r2_score(y, y_true)


###############################################################################
# Generate pipeline and train it
# ------------------------------
pipe = Pipeline(
    memory=None,
    steps=[
        (
            "variable_selection",
            GroupLasso(
                groups=groups,
                group_reg=20,
                l1_reg=0,
                scale_reg="inverse_group_size",
                subsampling_scheme=1,
                supress_warning=True,
            ),
        ),
        ("regressor", Ridge(alpha=0.1)),
    ],
)
pipe.fit(X, y)


###############################################################################
# Extract results and compute performance metrics
# -----------------------------------------------

# Extract from pipeline
yhat = pipe.predict(X)
sparsity_mask = pipe["variable_selection"].sparsity_mask
coef = pipe["regressor"].coef_.T

# Construct full coefficient vector
w_hat = np.zeros_like(w)
w_hat[sparsity_mask] = coef

R2 = r2_score(y, yhat)

# Print performance metrics
print(f"Number variables: {len(sparsity_mask)}")
print(f"Number of chosen variables: {sparsity_mask.sum()}")
print(f"R^2: {R2}, best possible R^2 = {R2_best}")


###############################################################################
# Visualise regression coefficients
# ---------------------------------
for i in range(w.shape[1]):
    plt.figure()
    plt.plot(w[:, i], ".", label="True weights")
    plt.plot(w_hat[:, i], ".", label="Estimated weights")

plt.figure()
plt.plot([w.min(), w.max()], [coef.min(), coef.max()], "gray")
plt.scatter(w, w_hat, s=10)
plt.ylabel("Learned coefficients")
plt.xlabel("True coefficients")
plt.show()
