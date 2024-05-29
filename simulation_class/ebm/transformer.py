# TODO docstrings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm, uniform
# https://www.andrewvillazon.com/custom-scikit-learn-transformers/
# https://datascience.stackexchange.com/questions/92968/why-use-fit-when-already-have-fit-transform

class ContinuousDistributionFitter(BaseEstimator, TransformerMixin):
    def __init__(self, distribution=norm, **dist_params):
        """
        Initialize the transformer with a specific distribution and its parameters.

        Parameters:
        distribution (scipy.stats distribution): The distribution to fit.
        dist_params (dict): Additional parameters for the distribution.
        """
        self.distribution = distribution
        self.dist_params = dist_params
        self.fitted_distributions = None

    def fit(self, X, y=None):
        """
        Fit the distribution to the data.
        """
        self.fitted_distributions = [self.distribution.fit(X[:, i], **self.dist_params) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        """
        Apply the fitted distribution to the data.

        Parameters:
        X (np.ndarray): The data to transform.
        
        Returns:
        np.ndarray: The transformed data.
        """
        transformed_data = np.zeros_like(X)
        for i in range(X.shape[1]):
            dist_params = self.fitted_distributions[i]
            transformed_data[:, i] = self.distribution(*dist_params).pdf(X[:, i])
        return transformed_data

    def fit_transform(self, X, y=None):
        """
        Fit the distribution to the data and transform it.
        """
        return self.fit(X, y).transform(X)
