'''
author: bg
goal: 
type: feature extraction transforms e.g. channels, coord-system, 
how: 
ref: 
refactors: 
'''

import utilz

from skimage import img_as_ubyte, img_as_float
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.decomposition import PCA 

from skimage.feature import local_binary_pattern

### TODO: remapped_data home/owner/mixin + save to file + skip/shortcuts use implications 
### ==== 1. Fundus  Color channelz ==== << TODO: beyond fundus 
class ColorChannelz(TransformerMixin, BaseEstimator):
    # def __init__(self):
    #     pass 

    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        return [self.remapped_data(x) for x in X] 

    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data
    def _get_channel_eq(self, img, c=-1): ## -1 is on gray scale 
        return utilz.Image.hist_eq( utilz.Image.get_channel(img, c) ) if c >= 0 \
            else utilz.Image.hist_eq( utilz.Image.gray_scale(img) )
      
    def remapped_data(self, img):  
        return self._get_channel_eq(img, 1) 
 
class FundusColorChannelz(ColorChannelz): 
    # def __init__(self):
    #     pass 
    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data 
    def _green_channel_update(self, img):
        return self._get_channel_eq(img, 1) 

    def _red_channel_update(self, img, thresh=0.97):
        o = self._get_channel_eq(img, 0) 
        rrange = o.max() - o.min() 
        o[ (o - o.min()/rrange) < thresh ] = 0        
        return o  

    def _blue_channel_update(self, img, thresh=1): 
        o = img_as_ubyte(img[:,:,-1].copy() )
        # print( o.min(), o.max() ) 
        o[ o!= thresh] = 0 #(thresh-o.min()+0.00001)/(o.max() - o.min())] = 0         
        o[ o == thresh] = 255   
        #o = utilz.Image.hist_eq( o )

        # o = self._get_channel_eq(img, 2)  
        # _omax = o.max() 
        # t = 1 if (thresh == 1 and o.max()==255) else (1/_omax) ##TODO: change o.max to dtype check + else case blue is lost;recompute thresh
        # o[ o != t] = 0
        return img_as_float(o) #.astype('uint8')

    def _vessels_channel(self, img, mtype=0):
        o = self._get_channel_eq(img, 2) ## on green channel 
        o = utilz.Image.edgez(o, mtype) #* 255
        return o #.astype('uint8')
    
    def remapped_data(self, img): 
        outiez = []
        # 1. resize, equalize, rescale-float :@: using self.clean_data 
        #img = utilz.Image.hist_eq(self.gray) 
        # 2. vessels
        outiez.append( self._vessels_channel(img ) )  
        # 3. color channelz
        outiez.append( self._green_channel_update( img )  ) 
        outiez.append( self._red_channel_update( img ) )
        outiez.append( self._blue_channel_update( img ) )   
        # 4. combine color channelz
        o = np.dstack(outiez) 
        return o 


class FundusAddLBP(ColorChannelz):

    def __init__(self, g_channel, lbp_radius = 1, lbp_method = 'uniform'):
        super(FundusAddLBP, self).__init__()  
        self.g_channel = g_channel 
        self.lbp_radius = lbp_radius 
        self.lbp_k = 8*lbp_radius
        self.lbp_method = lbp_method         

    def remapped_data(self, img):
        O_ = []
        o = img[:,:, self.g_channel]  
        o = local_binary_pattern( o, self.lbp_k, self.lbp_radius, self.lbp_method)
        
        c = img.shape[2]
        O_.append( o )
        O_ += [ img[:,:,i] for i in range(c)] 
        return np.dstack(O_) 

# Filter
class ChannelzSelector(TransformerMixin, BaseEstimator):
    def __init__(self, ls_channelz=(1,)):
        self.ls_channelz = ls_channelz 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [ x[:,:,self.ls_channelz] for x in X] 

### ==== 2. Eigenz ==== 
## EigenFeatures/Matrix Decomposition component selection based -- SVD, PCA, Selection 
## e.g. PCA using randomized SVD : https://scikit-learn.org/stable/modules/decomposition.html 
class EigenzChannelz(TransformerMixin, BaseEstimator): 
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

    def __init__(self, g_channel, topn, mtype=TYPE_PCA, mlevel=PER_CHANNEL, append_component=True):
        self.g_channel = g_channel 
        self.topn = topn 
        self.mtype = mtype 
        self.mlevel = mlevel 
        self.append_component = append_component 
        m, kargz = self._selectorz[mtype] 
        self.component_selector = m(n_components=topn, **kargz)

    def _get_op_channel(self, x):
        c = len(x[0].shape)
        if c <= 2:
            return x
        else:
            return [c[:,:,self.g_channel] for c in x] 

    def fit(self, X, y=None):
        ## first fit component_selector before transform 
        self.component_selector.fit( [self._get_op_channel(x) for x in X]  )  ## np.vectorize 
        return self 

    def transform(self, X, y=None):
        ## first fit component_selector before transform 
        return [self.remapped_data(x) for x in X ]   
        
    # per channel 
    def remapped_data(self, img): ## appends to the stack unless otherwise 
        ### TODO: per channel for now operating on one channel only but can append that to the original imag
        print("FIN-Egz: In: ", img.shape, " Out: ", o.shape ) e sent in
        if len(img.shape) <= 2:
            o = self.component_selector.transform(img) 
            return np.dstack([img, o]) if self.append_component else o 
        else:
            _,_, c = img.shape 
            o = self.component_selector.transform(self._get_op_channel(img) ) 
            return np.dstack([*[img[:,:,i] for i in range(c)], o]) if self.append_component else o  

    
### ==== 3. Patchify ==== <<< overlapping or not 
### TODO: refactor image dimensions calc  + Overlap size 
class PatchifyChannelz(TransformerMixin, BaseEstimator):
    def __init__(self, nx_patchez=9, origi_dim=(224, 224, -1), overlap_px = 10 ):
        self.nx_patchez = nx_patchez 
        self.origi_dim = origi_dim 
        self.overlap_px = overlap_px 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.remapped_data(x) for x in X]  

    def remapped_data(self, img):
        return utilz.Image.patchify_image(img, self.nx_patchez) 

