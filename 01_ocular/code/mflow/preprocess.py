'''
author: bg
goal: 
type: Data preprocessing tools = column/field/data cleaning transformerz
how: 
ref: 
refactors: 
'''

import  utilz 

from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.preprocessing import FunctionTransformer 


from sklearn.compose import TransformedTargetRegressor  ## TODO: for ylabelz  ++ one-hot-encoding, 



### 1. flatten data 
# TODO: numpy.ravel, since it doesn't make a copy of the array, but just return a view of the array, which is much faster than numpy.flatten. 
class Flattenor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ = [x.ravel() for x in X] 
        print( "Flattenor -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        return O_
## 2. reshapes e.g. (1, xxx), or from flat to 2D etc 
class Reshapeor(TransformerMixin, BaseEstimator):
    def __init__(self, newshape):
        self.newshape = newshape         
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ = [x.reshape( self.newshape ) for x in X] 
        print( "3D -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        return O_

## 3. load from file 
class LoadImageFileTransform(TransformerMixin, BaseEstimator):
    def __init__(self, fpath_colname, resize=(224,224)): 
        self.fpath_colname = fpath_colname 
        self.resize = resize 
    def fit(self, X, y=None):
        return self    
    def transform(self, X, y=None): ##TODO: memory mgt 
        fpathz = X.loc[:, self.fpath_colname ] ## from pdframe 
        O_ = []
        for fp in fpathz:
            img = self.get_image_from_file(fp)  
            O_.append( img ) 
        return O_      
    def get_image_from_file(self, fp):
        return utilz.Image.fetch_and_resize_image(fp, self.resize)  
