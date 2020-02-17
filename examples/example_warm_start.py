"""
Warm start to choose regularisation strength
============================================
"""

###############################################################################
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np

from group_lasso import GroupLasso

np.random.seed(0)
GroupLasso.LOG_LOSSES = True


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
# Generate estimator and train it
# -------------------------------
num_regs = 10
regularisations = np.logspace(-0.5, 1.5, num_regs)
weights = np.empty((num_regs, w.shape[0],))
gl = GroupLasso(
    groups=groups,
    group_reg=5,
    l1_reg=0,
    frobenius_lipschitz=True,
    scale_reg="inverse_group_size",
    subsampling_scheme=1,
    supress_warning=True,
    n_iter=1000,
    tol=1e-3,
    warm_start=True,  # Warm start to start each subsequent fit with previous weights
)

for i, group_reg in enumerate(regularisations[::-1]):
    gl.group_reg = group_reg
    gl.fit(X, y)
    weights[-(i + 1)] = gl.sparsity_mask_.squeeze()


###############################################################################
# Visualise chosen covariate groups
# ---------------------------------
plt.figure()
plt.pcolormesh(np.arange(w.shape[0]), regularisations, -weights, cmap="gray")
plt.yscale("log")
plt.xlabel("Covariate number")
plt.ylabel("Regularisation strength")
plt.title("Active groups are black and inactive groups are white")
plt.show()
