from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """embarked column imputer."""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna( self.fill_value)

        return X

class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X

class tempOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables):

      if not isinstance(variables, str):
        raise ValueError('variables should be a str')
      self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
      q1 = X.describe()[self.variables].loc['25%']
      q3 = X.describe()[self.variables].loc['75%']
      iqr = q3 - q1
      lower_bound = q1 - (1.5 * iqr)
      upper_bound = q3 + (1.5 * iqr)
      self.lbound = X[self.variables].mean()
      self.ubound = X[self.variables].std()
      # we need this step to fit the sklearn pipeline
      return self

    def transform(self, X):
      X = X.copy() # so that we do not over-write the original dataframe
      for i in X.index:
        if X.loc[i,self.variables] > self.ubound:
            X.loc[i,self.variables]= self.ubound
        if X.loc[i,self.variables] < self.lbound:
            X.loc[i,self.variables]= self.lbound
      return X


class windspeedOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables):

      if not isinstance(variables, str):
        raise ValueError('variables should be a str')
      self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
      q1 = X.describe()[self.variables].loc['25%']
      q3 = X.describe()[self.variables].loc['75%']
      iqr = q3 - q1
      lower_bound = q1 - (1.5 * iqr)
      upper_bound = q3 + (1.5 * iqr)
      self.lbound = X[self.variables].mean()
      self.ubound = X[self.variables].std()
      # we need this step to fit the sklearn pipeline
      return self

    def transform(self, X):
      X = X.copy() # so that we do not over-write the original dataframe
      for i in X.index:
        if X.loc[i,self.variables] > self.ubound:
            X.loc[i,self.variables]= self.ubound
        if X.loc[i,self.variables] < self.lbound:
            X.loc[i,self.variables]= self.lbound
      return X

class humOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables):

      if not isinstance(variables, str):
        raise ValueError('variables should be a str')
      self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
      q1 = X.describe()[self.variables].loc['25%']
      q3 = X.describe()[self.variables].loc['75%']
      iqr = q3 - q1
      lower_bound = q1 - (1.5 * iqr)
      upper_bound = q3 + (1.5 * iqr)
      self.lbound = X[self.variables].mean()
      self.ubound = X[self.variables].std()
      # we need this step to fit the sklearn pipeline
      return self

    def transform(self, X):
      X = X.copy() # so that we do not over-write the original dataframe
      for i in X.index:
        if X.loc[i,self.variables] > self.ubound:
            X.loc[i,self.variables]= self.ubound
        if X.loc[i,self.variables] < self.lbound:
            X.loc[i,self.variables]= self.lbound
      return X

