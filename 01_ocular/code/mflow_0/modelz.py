'''
author: bg
goal: 
type: Model estimators  
how: sklearn as manager (pipeline) + Pytorch nn + keras@TL weights/encoders 
ref: 
refactors: 
'''

import numpy as  np 
import torch 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor  


from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA