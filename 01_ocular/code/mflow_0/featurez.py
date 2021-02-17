
import utilz
from zdata import RemappedFundusImage 

import numpy as  np 
import pandas as pd 
import torch 
import sklearn  
from sklearn.base import BaseEstimator, TransformerMixin 

from sklearn.pipeline import Pipeline 

from sklearn.decomposition import PCA

class ZFeatureMapper:

    def remapped_data(self, img):
        return img

    def save_fmap(self, fpath):
        pass ## TODO: how to use the serializable obj 


## === Color channelz
class ColorChannelzRemap(ZFeatureMapper):
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data
    def _get_channel_eq(self, img, c):
        return utilz.Image.hist_eq( utilz.Image.get_channel(img, c) ) 

    def green_channel_update(self, img):
        o = self._get_channel_eq(img, 1) 
        return o 

    def red_channel_update(self, img, thresh=0.97):
        o = self._get_channel_eq(img, 0) 
        rrange = o.max() - o.min() 
        o[ (o - o.min()/rrange) < thresh ] = 0
        return o 

    def blue_channel_update(self, img, thresh=1): 
        o = self._get_channel_eq(img, 2)  
        t = 1 if (thresh == 1 and o.max()==255) else (1/255) ##TODO: change o.max to dtype check + else case blue is lost;recompute thresh
        o[ o != t] = 0
        return o

    def vessels_channel(self, img, mtype=2):
        o = self._get_channel_eq(img, 2) 
        o = utilz.Image.edgez(o, mtype) 
        return o
    
    def remapped_data(self, img): 
        outiez = []
        # 1. resize, equalize, rescale-float :@: using self.clean_data 
        #img = utilz.Image.hist_eq(self.gray) 
        # 2. vessels
        outiez.append( self.vessels_channel(img ) )  
        # 3. color channelz
        outiez.append( self.green_channel_update( img )  ) 
        outiez.append( self.red_channel_update( img ) )
        outiez.append( self.blue_channel_update( img ) )   
        # 4. combine color channelz
        o = np.dstack(outiez) 
        return o 
    

## === EigenFeatures/Matrix Decomposition component selection based -- SVD, PCA, Selection 
## e.g. PCA using randomized SVD : https://scikit-learn.org/stable/modules/decomposition.html 
class ComponentSelectionRemap(ZFeatureMapper):
    ## TODO: Menu correctness and indexing + Pipeline @ use SKLearn decompositions 
    TYPE_PCA = 0
    TYPE_SELECT = 1
    TYPE_PCA_SELECT = 2
    TYPE_LCA = 3
    ## Per Channel Vs Overall Vs Both  TODO: correctness and indexing 
    PER_CHANNEL = 0
    PER_FULLIMG = 1
    PER_CHANNEL_FULLIMG = 2 

    _selectorz = [
        (PCA, {'svd_solver': 'randomized', 'whiten':True} ) 
    ]

    def __init__(self, topn, mtype=TYPE_PCA, mlevel=PER_CHANNEL):
        self.topn = topn 
        self.mtype = mtype 
        self.mlevel = mlevel 
        m, kargz = self._selectorz[mtype] 
        self.trans_pipeline = m(n_components=topn, **kargz)

    # per channel 
    def remapped_data(self, img): 
        x, y, c = img.shape 
        nx, ny = self.topn, -1 #int(np.sqrt(self.topn)), -1 ##TODO: check compute
        def trans(img):
            print("Egz: In: ", img.shape, " Goal: ", (nx, ny), " of ", self.topn )  
            t = self.trans_pipeline.fit_transform(img )  # .flatten()
            t = t.reshape( (nx, ny) ) ## (topn, h, w) 
            return t 

        o = img.copy() 
        if len(img.shape) == 2 or self.mlevel == ComponentSelectionRemap.PER_FULLIMG:
            o = trans(img) 
        else: 
            O_ = []
            for i in range(c):
                O_.append( trans(img[:,:,i] ) ) 
            
            if self.mlevel == ComponentSelectionRemap.PER_FULLIMG:
                O_.append( trans(img.flatten()) ) 
            o = np.dstack( O_ ) 
        print("FIN-Egz: In: ", img.shape, " Out: ", o.shape ) 
        return o 

    