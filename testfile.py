"""
Comparison of standard logistic regression
"""

import numpy as np
import matplotlib.pyplot as plt
from group_lasso import LogisticGroupLasso

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

################################################################################
def compute_logistic_regression(X, y, rs, n_iter=100):
    """
    Note: sklearn's LogisticRegression by default applies regularization.
    In the experiments below, scores are lower with regularization (less
    overfitting), however the effect should not be big because d=2 << n=200.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    lr = LogisticRegression(
        random_state=rs, solver="sag", max_iter=n_iter, tol=1e-4,
    )
    lr.fit(X, y)
    y_pred = lr.predict(X)
    accuracy = (y_pred == y).mean()
    score = lr.score(X, y)
    assert accuracy == score
    return score


################################################################################
def compute_logistic_group_lasso(X, y, rs, n_iter=50):
    LogisticGroupLasso.LOG_LOSSES = True
    gl = LogisticGroupLasso(
        groups=np.arange(X.shape[1]),
        group_reg=0.0,
        l1_reg=0.0,
        supress_warning=True,
        random_state=rs,
        n_iter=n_iter,
        tol=1e-4,
    )
    gl.fit(X, y)
    y_pred = gl.predict(X).ravel()
    accuracy = (y_pred == y).mean()
    score = gl.score(X, y)
    assert accuracy == score
    return score


################################################################################
def generate_classification(n, d, noise, rs):
    """
    n:      number of samples per class
    d:      number of features
    noise:  amount of uncertainty added to
    rs:     random state

    Balanced classes, noise governed by a normal distribution,
    """
    # Create separable centers.
    dist = 10
    mu1 = rs.uniform(high=10, size=d)
    mu2 = rs.uniform(low=-1, high=1, size=d)  # use as direction
    mu2 = mu2 / np.linalg.norm(mu2)
    mu2 = mu1 + dist * mu2

    # Noise
    n1 = n
    n2 = n
    X1 = rs.normal(mu1, noise, size=(n1, d))
    X2 = rs.normal(mu2, noise, size=(n2, d))
    y1 = np.ones(n1)
    y2 = np.zeros(n2)

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2])

    # Shuffle
    i = rs.permutation(len(y))
    X = X[i, :]
    y = y[i]
    return X, y


################################################################################
# MAIN
################################################################################

# Compare LogisticGroupLasso with sklearn's LogisticRegression...
#   1) ... on synthetic data
#   2) ... on real-world data (the breast-cancer dataset)

rs = np.random.RandomState(42)

###########################################################
# Synthetic data
###########################################################
print("1) Synthethic problem...")

###########################################################
# Settings.
d = 2  # Number of features
n = 100  # Number of samples per class
p = 100  # Number of difficulty levels
n_iter = 50  # Number of max iterations
###########################################################
#%%
noises = np.linspace(1, 10, p)
difficulty = (noises - min(noises)) / (max(noises) - min(noises))
scores_lr = np.zeros(p)
scores_gl = np.zeros(p)
for i, noise in enumerate(noises):
    print("%d/%d" % (i + 1, len(noises)))

    X, y = generate_classification(n=n, d=d, noise=noise, rs=rs)

    scores_lr[i] = compute_logistic_regression(X, y, rs)
    scores_gl[i] = compute_logistic_group_lasso(X, y, rs, n_iter=n_iter)

plt.style.use("seaborn-pastel")
plt.plot(difficulty, scores_lr, label="sklearn (n_iter=100)")
plt.plot(difficulty, scores_gl, label="group-lasso (n_iter=%s)" % n_iter)
plt.legend()
plt.ylim([-0.05, 1.05])
plt.xlabel("noise level - problem difficulty")
plt.ylabel("prediction accuracy")
plt.title("Comparison of standard logistic regression")
txt = "Average diff: %.2f" % np.mean(scores_lr - scores_gl)
plt.annotate(
    txt, xy=(0.9, 0.1), va="top", ha="right", xycoords="axes fraction"
)
plt.plot([min(difficulty), max(difficulty)], [0.5, 0.5], ":k", alpha=0.6)
plt.savefig(
    f"comparison-niter={n_iter}.pdf", transparent=False, bbox_inches="tight"
)

###########################################################
# 2) Real-world problem
###########################################################
#%%

# Real-world problem.
print()
print("2) Real world problem...")
X, y = load_breast_cancer(return_X_y=True)

gl = LogisticGroupLasso(
    groups=np.arange(X.shape[1]),
    group_reg=0.0,
    l1_reg=0.0,
    supress_warning=True,
    random_state=rs,
    n_iter=n_iter,
    tol=1e-4,
)
gl.fit(X, y)
y_pred = gl.predict(X).ravel()
score_gl = (y_pred == y).mean()

score_lr = compute_logistic_regression(X, y, rs)

print("sklearn:    ", score_lr)  # 0.9472759226713533, lbfgs not converged
print("group-lasso:", score_gl)  # 0.8910369068541301, FISTA not converged
print()

plt.show()
