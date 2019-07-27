Installation guide
==================

Dependencies
------------

``group-lasso`` support Python 3.5+, Additionally, you will need ``numpy``,
and ``scikit-learn`` (which again requires ``scipy`` and ``joblib``). These
packages should come pre-installed on any Anaconda installation, otherwise,
they can be installed using ``pip``::

    pip install numpy
    pip install scikit-learn

Installing group-lasso
----------------------

``group-lasso`` is available through Pypi and can easily be installed with a
pip install::

    pip install group-lasso

I update the Pypi version regularly, however for the latest update, you should
clone from GitHub and install it directly, as so::

    git clone https://github.com/yngvem/group-lasso.git
    cd group-lasso
    python setup.py
