'''
author: bg
goal: 
type: feature extraction transforms e.g. channels, coord-system, 
how: 
ref: 
refactors: 
'''

import utilz

from sklearn.base import TransformerMixin, BaseEstimator 

### TODO: remapped_data home/owner/mixin + save to file + skip/shortcuts use implications 
### ==== 1. Fundus  Color channelz ==== << TODO: beyond fundus 
class FundusColorChannelz(TransformerMixin, BaseEstimator):
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

    def _green_channel_update(self, img):
        return self._get_channel_eq(img, 1) 

    def _red_channel_update(self, img, thresh=0.97):
        o = self._get_channel_eq(img, 0) 
        rrange = o.max() - o.min() 
        o[ (o - o.min()/rrange) < thresh ] = 0
        return o 

    def _blue_channel_update(self, img, thresh=1): 
        o = self._get_channel_eq(img, 2)  
        t = 1 if (thresh == 1 and o.max()==255) else (1/255) ##TODO: change o.max to dtype check + else case blue is lost;recompute thresh
        o[ o != t] = 0
        return o

    def _vessels_channel(self, img, mtype=2):
        o = self._get_channel_eq(img, 2) ## on green channel 
        o = utilz.Image.edgez(o, mtype) 
        return o
    
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

    def __init__(self, topn, mtype=TYPE_PCA, mlevel=PER_CHANNEL):
        self.topn = topn 
        self.mtype = mtype 
        self.mlevel = mlevel 
        m, kargz = self._selectorz[mtype] 
        self.component_selector = m(n_components=topn, **kargz)

    def fit(self, X, y=None):
        ## first fit component_selector before transform 
        self.component_selector.fit( X ) 
        return self 

    def transform(self, X, y=None):
        ## first fit component_selector before transform 
        return [self.remapped_data(x) for x in X]  
        
    # per channel 
    def remapped_data(self, img):         
        x, y, c = img.shape 
        nx, ny = self.topn, -1 #int(np.sqrt(self.topn)), -1 ##TODO: check compute
        def trans(img):
            print("Egz: In: ", img.shape, " Goal: ", (nx, ny), " of ", self.topn )  
            t = self.component_selector.transform(img )  # .flatten() ### TODO: fit on who and then only transform per 
            t = t.reshape( (nx, ny) ) ## (topn, h, w)  ##TODO: sense check 
            return t 
        ## do per channel seperately or not TODO: sense check flow 
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
        x, y, c = img.shape  
        padded_wx = ((x//self.nx_patchez)*self.nx_patchez)  + self.nx_patchez 
        padded_hy = ((y//self.nx_patchez)*self.nx_patchez) + self.nx_patchez 

        oimg = np.zeros( (padded_wx, padded_hy, c) ) 
        print(oimg.shape) 

        oimg[:x, :y, :] = img 

        O_ = []
        px = padded_wx//self.nx_patchez
        py = padded_hy//self.nx_patchez
        for i in range(self.nx_patchez): 
            for j in range(self.nx_patchez):                 
                O_.append( oimg[ (i*px):((i+1)*px), (j*py):((j+1)*py), :] ) 
        O_ = np.dstack(O_)  
        print('patch.dim: ', O_.shape )
        
        return O_ 

