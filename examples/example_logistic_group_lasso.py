"""
GroupLasso for logistic regression
==================================

A sample script for group lasso regression
"""

###############################################################################
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from group_lasso import LogisticGroupLasso

np.random.seed(0)
LogisticGroupLasso.LOG_LOSSES = True


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
noise_std = 1


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
p = 1 / (1 + np.exp(-y))
p_true = 1 / (1 + np.exp(-y_true))
c = np.random.binomial(1, p_true)


###############################################################################
# View noisy data and compute maximum accuracy
# --------------------------------------------
plt.figure()
plt.plot(p, p_true, ".")
plt.xlabel("Noisy probabilities")
plt.ylabel("Noise-free probabilities")
# Use noisy y as true because that is what we would have access
# to in a real-life setting.
best_accuracy = ((p_true > 0.5) == c).mean()


###############################################################################
# Generate estimator and train it
# -------------------------------
gl = LogisticGroupLasso(
    groups=groups,
    group_reg=0.05,
    l1_reg=0,
    scale_reg="inverse_group_size",
    subsampling_scheme=1,
    supress_warning=True,
)

gl.fit(X, c)


###############################################################################
# Extract results and compute performance metrics
# -----------------------------------------------

# Extract info from estimator
pred_c = gl.predict(X)
sparsity_mask = gl.sparsity_mask
w_hat = gl.coef_

# Compute performance metrics
accuracy = (pred_c == c).mean()

# Print results
print(f"Number variables: {len(sparsity_mask)}")
print(f"Number of chosen variables: {sparsity_mask.sum()}")
print(f"Accuracy: {accuracy}, best possible accuracy = {best_accuracy}")


###############################################################################
# Visualise regression coefficients
# ---------------------------------
for i in range(w.shape[1]):
    plt.figure()
    plt.plot(w[:, i] / np.linalg.norm(w[:, i]), ".", label="True weights")
    plt.plot(
        gl.coef_[:, i] / np.linalg.norm(gl.coef_[:, i]),
        ".",
        label="Estimated weights",
    )

plt.figure()
plt.plot([w.min(), w.max()], [gl.coef_.min(), gl.coef_.max()], "gray")
plt.scatter(w, gl.coef_, s=10)
plt.ylabel("Learned coefficients")
plt.xlabel("True coefficients")

plt.figure()
plt.plot(gl.losses_)

plt.show()
