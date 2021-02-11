'''
author: bg
goal: 
type: Data transformers  -- data preproc and handcrafted features 
how: sklearn pipeline management 
ref: 
refactors: 
'''

import utilz
from zdata import RemappedFundusImage 

import numpy as  np 
import pandas as pd 
import torch 
import sklearn 
from sklearn.compose import TransformedTargetRegressor  
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import Pipeline 

from sklearn.decomposition import PCA
#from sklearn.decomposition import 

# this is not necc TODO: refactor 
class STReshapeFlatten:
    def __init__(self, reshape = False, flatten=True):
        self.flatten = flatten 
        self.reshape = reshape 
    
    def reformat(self, img):
        img_ = img.copy() ## neccc???
        if self.flatten:
            img_ = img_.flatten()
        if self.reshape: 
            img_.reshape( (1, *img.shape) )
        return img_

## TODO: refactor at above 
class Flattenor(STReshapeFlatten, TransformerMixin, BaseEstimator):
    def __init__(self, reshape = False, flatten=True):
        super().__init__(reshape, flatten) 
        self.flatten = flatten 
        self.reshape = reshape 
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ = [self.reformat(x) for x in X] 
        print( "Flattenor -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        return O_

class ReshapeToImageD(TransformerMixin, BaseEstimator):
    def __init__(self, imsize, reshape = False):
        self.imsize = imsize  
        self.reshape = reshape 
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ = [x.reshape( (*self.imsize, -1) ) for x in X] 
        print( "3D -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        return O_

# load from file and flatten @ generic
class LoadImageFileTransform(STReshapeFlatten, TransformerMixin, BaseEstimator):
    def __init__(self, fpath_colname, size=(224,224), reshape = False, flatten=True):
        super().__init__(reshape, flatten) 
        self.fpath_colname = fpath_colname 
        self.size = size 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None): ##TODO: memory mgt 
        fpathz = X.loc[:, self.fpath_colname ] ## from pdframe 
        X_ = []
        for fp in fpathz:
            img = self.get_image_from_file(fp) 
            img = self.reformat(img) 
            X_.append( img ) 
        return X_ 
    
    def get_image_from_file(self, fp):
        return utilz.Image.fetch_and_resize_image(fp, self.size) 


class FundusFeatureMapTransform(LoadImageFileTransform):
    def __init__(self, fpath_colname, size=(224,224), reshape = False, flatten=True):
        super().__init__(fpath_colname, size=size,  reshape = reshape, flatten=flatten)  

    def get_image_from_file(self, fp):
        img = RemappedFundusImage('img', fp, resize_dim=self.size) 
        return img.remapped_data 

### TODO: refactor image dimensions calc 
class PatchifyTransform(STReshapeFlatten, TransformerMixin, BaseEstimator):
    def __init__(self, nx_patchez=9, origi_dim=(224, 224, -1), reshape = False, flatten=True):
        super().__init__(reshape = reshape, flatten=flatten)   
        self.nx_patchez = nx_patchez 
        self.origi_dim = origi_dim 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = []   
        for img in X: 
            # img = img.reshape(self.origi_dim)
            x, y, c = img.shape  
            pwx = ((x//self.nx_patchez)*self.nx_patchez)  + self.nx_patchez 
            phy = ((y//self.nx_patchez)*self.nx_patchez) + self.nx_patchez 

            oimg = np.zeros( (pwx, phy, c) ) 
            print(oimg.shape) 

            oimg[:x, :y, :] = img 
            O_ = []
            px = pwx//self.nx_patchez
            py = phy//self.nx_patchez
            for i in range(self.nx_patchez): 
                for j in range(self.nx_patchez):                 
                    O_.append( oimg[ (i*px):((i+1)*px), (j*py):((j+1)*py), :] ) 
            O_ = np.dstack(O_) 
            X_.append(O_)
            print('patch.dim: ', O_.shape )
        return X_ 
    


if __name__ == '__main__': 
    fp1 = '/mnt/externz/zRepoz/datasets/fundus/stare/im0064.ppm'
    fp2 = '/mnt/externz/zRepoz/datasets/fundus/stare/im0264.ppm'
    df = pd.DataFrame.from_records([[fp1, 'The quick brown fox jumped over the lazy dogs'],
                                     [fp2, "yet another image here"]
                                    ],columns=['fpath', 'extra'])
    nx_patchez = 3
    patch_dim = (75,75)
    origi_dim = (224, 224)

    gopatch = True  

    piper = Pipeline([('fmapper', FundusFeatureMapTransform(fpath_colname='fpath', reshape = False, flatten=False)),  ##pd frame 
                        ('patchie', PatchifyTransform( nx_patchez=nx_patchez, origi_dim=origi_dim)), ## img array/list 
                        ('flatten', Flattenor()), ## this is repeated as a standalone TODO: refactor at class level
                        ('scaler', StandardScaler() ),
                        ('re-imgD', ReshapeToImageD( patch_dim if gopatch else origi_dim ) ) ]) ## TODO: compute/get patch_dim: (75, 75)

    outiez = piper.fit_transform(df)[0]


    if gopatch:
        x, y, c = outiez.shape 
        pw = c//nx_patchez**2 
        print(outiez.shape, c, pw , nx_patchez)
        _ = [print( (i*pw), " to ", ((i+1)*pw) ) for i in range(nx_patchez**2)]
        
        utilz.Image.plot_images_list([outiez[:,:,(i*pw):((i+1)*pw) ][:,:,:3] for i in range(nx_patchez**2)], nc=3, cmap='gray')  #int(np.sqrt(nx_patchez))
        for cj in range( pw ):
            utilz.Image.plot_images_list([outiez[:,:,(i*pw):((i+1)*pw) ][:,:,cj].reshape( patch_dim ) for i in range(nx_patchez**2)], nc=3, cmap='gray') #np.sqrt(nx_patchez)

    else:
        co = outiez.shape[2]
        rp = -1/outiez[:,:,0:3] #* outiez[:,:,3]
        utilz.Image.plot_images_list( [-outiez[:,:,i] for i in range(co)]+[rp,],nc=co+1, cmap='gray')  
    

