'''
author: bg
goal: 
type: Data transformers  -- data preproc and handcrafted features 
how: sklearn pipeline management 
ref: 
refactors: 
'''

import utilz, featurez 
from zdata import RemappedFundusImage 

import numpy as  np 
import pandas as pd 

import matplotlib.pyplot as plt 

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


class FundusFeatureMapTransform(featurez.ColorChannelzRemap,LoadImageFileTransform):
    def __init__(self, fpath_colname, size=(224,224), reshape = False, flatten=True):
        super(FundusFeatureMapTransform, self).__init__(fpath_colname, size=size,  reshape = reshape, flatten=flatten)  

    def get_image_from_file(self, fp):
        #img = RemappedFundusImage('img', fp, resize_dim=self.size) 
        o = LoadImageFileTransform.get_image_from_file(self, fp) 
        print( "FETCHED: ", o.shape )
        o = featurez.ColorChannelzRemap.remapped_data( self, o )
        return o 

class EigenzMapTransform(featurez.ComponentSelectionRemap,LoadImageFileTransform):
    ## TODO: fix init cascade 
    def __init__(self, fpath_colname, topn, size=(224,224), reshape = False, flatten=True):
        LoadImageFileTransform.__init__(self, fpath_colname, size=size,  reshape = reshape, flatten=flatten)  
        featurez.ComponentSelectionRemap.__init__(self, topn)
        self.topn = topn 

    def get_image_from_file(self, fp):
        #img = RemappedFundusImage('img', fp, resize_dim=self.size) 
        o = LoadImageFileTransform.get_image_from_file(self, fp) 
        print( "Eigenz FETCHED: ", o.shape )
        o = featurez.ComponentSelectionRemap.remapped_data( self, o )
        print( "Eigenz: ", o.shape )
        return o 

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
    

    ## --- Setup ylabelz based on DShortCode <<< TODO:

# class YLabelzFundusTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, dscode='dscodez_short', healthycol='Normal', undefcol='UNDEF' ): 
#         self.healthycol = healthycol
#         self.undefcol = undefcol
#         self.dscodez = dscode 
    
#     def fit(self, X, y=None):
#         return self
    
    
#     def transform(self, X, y=None):
#         X_ = X.copy()
#         X_['dclass'] = 'UNDEF' 
#         X_['dclass_v'] = 0 
#         X_.loc[ ( X_[self.healthycol] > 0 ), ['dclass']] = 'Normal' 
#         X_.loc[ ( X_[self.healthycol] > 0 ), ['dclass_v']] = 1 
#         X_.loc[ (X_[self.healthycol] <= 0) & (X_[self.undefcol] <= 0), ['dclass']] = 'Sick' 
#         X_.loc[ (X_[self.healthycol] <= 0) & (X_[self.undefcol] <= 0), ['dclass_v']] = 2
        
# #         X_['dclass'] = 'Not Sick' 
# #         X_['dclass_v'] = 0  
# #         X_.loc[ (X_[self.healthycol] != 1) & (X_[self.undefcol] != 1), ['dclass']] = 'Sick' 
# #         X_.loc[ (X_[self.healthycol] != 1) & (X_[self.undefcol] != 1), ['dclass_v']] = 1
#         #print(  )
#         return X_
    
    


if __name__ == '__main__': 
    fp1 = '/mnt/externz/zRepoz/datasets/fundus/stare/im0064.ppm'
    fp2 = '/mnt/externz/zRepoz/datasets/fundus/stare/im0264.ppm'
    df = pd.DataFrame.from_records([[fp1, 'The quick brown fox jumped over the lazy dogs'],
                                     [fp2, "yet another image here"]
                                    ],columns=['fpath', 'extra'])
    nx_patchez = 7
    nc = 3   #int(np.sqrt(nx_patchez)) 
    origi_dim = (224, 224)
    ipd = int(  (origi_dim[0]//nx_patchez)*nx_patchez + nx_patchez ) 
    patch_dim = (ipd, ipd) #(75,75)
    pca_topn = 221 

    gopatch = False  

    piper = Pipeline([('fmapper', FundusFeatureMapTransform(fpath_colname='fpath', reshape = False, flatten=False)),  ##pd frame 
                        # ('patchie', PatchifyTransform( nx_patchez=nx_patchez, origi_dim=origi_dim)), ## img array/list 
                        ('flatten', Flattenor()), ## this is repeated as a standalone TODO: refactor at class level
                        ('scaler', StandardScaler() ),
                        ('re-imgD', ReshapeToImageD( patch_dim if gopatch else origi_dim ) ) ]) ## TODO: compute/get patch_dim: (75, 75)

    outiez = piper.fit_transform(df)[0]


    if gopatch:
        x, y, c = outiez.shape 
        pw = c//nx_patchez**2 
        print(outiez.shape, c, pw , nx_patchez)
        _ = [print( (i*pw), " to ", ((i+1)*pw) ) for i in range(nx_patchez**2)]
        
        utilz.Image.plot_images_list([outiez[:,:,(i*pw):((i+1)*pw) ][:,:,:3] for i in range(nx_patchez**2)], nc=nc, cmap='gray') 
        for cj in range( pw ):
            utilz.Image.plot_images_list([outiez[:,:,(i*pw):((i+1)*pw) ][:,:,cj].reshape( patch_dim ) for i in range(nx_patchez**2)], nc=nc, cmap='gray')  

    else:
        co = outiez.shape[2]
        rp = -1/outiez[:,:,0:3] #* outiez[:,:,3]
        utilz.Image.plot_images_list( [outiez[:,:,i] for i in range(co)]+[rp,],nc=co+1, cmap='gray')  
     
    
    piper = Pipeline([('fmapper', EigenzMapTransform(fpath_colname='fpath', topn=pca_topn, reshape = False, flatten=False)),  ##pd frame 
                        # ('patchie', PatchifyTransform( nx_patchez=nx_patchez, origi_dim=origi_dim)), ## img array/list 
                        # ('pca', featurez.ComponentSelectionRemap() ), 
                        ('flatten', Flattenor()), ## this is repeated as a standalone TODO: refactor at class level
                        ('scaler', StandardScaler() ),
                        ('re-imgD', ReshapeToImageD( (pca_topn, 224) ) ) ]) ## TODO: compute/get patch_dim: (75, 75)

    outiez = piper.fit_transform(df)[0]
    co = outiez.shape[2]
    rp = -1/outiez[:,:,0:3] #* outiez[:,:,3]
    utilz.Image.plot_images_list( [outiez[:,:,i] for i in range(co)]+[rp,],nc=co+1, cmap=plt.cm.RdBu)  

    