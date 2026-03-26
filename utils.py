"""
Shared utilities for the CYP11B2/B1 QSAR pipeline.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IndexSelector(BaseEstimator, TransformerMixin):
    """
    Select a fixed subset of columns from a feature matrix.

    Designed to transform X_combined (2189d) → X_selected (300d)
    using pre-computed column indices from step2b_feature_select.py.

    Parameters
    ----------
    indices : array-like of int
        Column indices to keep.

    Example
    -------
    selector = joblib.load("models/feature_selector.pkl")
    X_sel = selector.transform(X_combined)   # (n, 300)
    """

    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices]
