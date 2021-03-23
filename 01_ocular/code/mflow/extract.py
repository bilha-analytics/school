'''
author: bg
goal: 
type: feature extraction transforms e.g. channels, coord-system, 
how: 
ref: 
refactors: 
'''

import utilz

from skimage import img_as_ubyte, img_as_float, img_as_uint
from skimage import color as sk_color
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
    def _get_channel_eq(self, img, c=-1, eq_mtype=1): ## -1 is on gray scale 
        return utilz.Image.hist_eq( utilz.Image.get_channel(img, c), mtype=eq_mtype ) if c >= 0 \
            else utilz.Image.hist_eq( utilz.Image.gray_scale(img), mtype=eq_mtype  )
    
    def _get_lab_img(self, img, extractive=True): 
        o = sk_color.rgb2lab( img ) 
        if extractive:
            l = o[:,:,0]
            ## bluez are -ves, redz are positivez
            b = o[:,:,-1]
            b[ b >=0 ] = 0
            ## QUE: add back yellow to red or not ?? << does it seem to be useful << TODO: review
            r = o[:,:,1]
            # y = r.copy()
            # y[y>=0] = 0
            # process red
            r[r <= 0 ] = 0
            # add back yellow
            # r = r*y 
            o = np.dstack([l,r,b])  
        # normalize-ish :/ <<< TODO: fix 
        abs_max = np.max( np.abs( o ) ) 
        o = o/abs_max
        return o 

    ## TODO: Thresholding
    def _get_yellow_from_rgb2lab(self, img):
        o = sk_color.rgb2lab( img )  
        ##yellow is in A and is -ves
        o = o[:,:,1]
        o[o >= 0 ] = 0
        o = -1 * o 
        # normalize-ish :/ <<< TODO: fix 
        abs_max = np.max( np.abs( o ) ) 
        o = o/abs_max
        return o 

    def _lab_to_rgb(self, img):
        return sk_color.lab2rgb( img ) 

    def _get_color_eq(self, img):
        ## rgb to lab --> equalize l --> lab to rgb  <<< TODO: move to utilz         
        ## REF: d-hazing and underwater images 
        # 1. LAB color space intensity Vs luminous 
        # a. rgb2lab -> clahe(l) -> lab2rgb
        # o = sk_color.rgb2lab( img ) 
        # eq_l = self._get_channel_eq( img_as_uint(o), 0, eq_mtype=0) 
        # o = np.dstack( [eq_l, o[:,:,1], o[:,:,2]])
        # ## b. rgb2gray -> gradient smooth -> gray to rgb 
        # o = sk_color.lab2rgb( o ) 

        ## 2. CLAHE/CStreching per channel 
        o = [self._get_channel_eq(img, i, eq_mtype=1) for i in range(3)] 
        o = np.dstack(o) 

        return o

    def remapped_data(self, img):  
        return self._get_channel_eq(img, 1) 
 
### TODO: at choice of cleaning
class OrigiCleanedChannelz(ColorChannelz): 
    def remapped_data(self, img):  ## simply clahe something something 
        o = [self._get_channel_eq(img, i) for i in range(3)] 
        o = np.dstack(o) 
        # if len(img.shape) <= 2:
        #     return np.dstack([img, o]) if self.append_component else o 
        # else:
        #     return np.dstack([*[img[:,:,i] for i in range(c)], o]) if self.append_component else o  

        return self._get_color_eq(img)

class FundusColorChannelz(ColorChannelz):  
    
    def __init__(self, add_origi=True , red_thresh=0.97, color_space='rgb'):
        super(FundusColorChannelz, self).__init__()  
        self.add_origi = add_origi   
        self.red_thresh = red_thresh 
        self.color_space = color_space 
    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data 
    def _green_channel_update(self, img):
        return self._get_channel_eq(img, 1) 

    def _red_channel_update(self, img, thresh=0.97):
        # if self.color_space == 'lab':
        #     ## redz are positives 
        #     o = self._get_channel_eq(img, 1) 
        #     rrange = o.max() - o.min() 
        #     o[ (o - o.min()/rrange) < thresh ] = 0   
        # else:
        o = self._get_channel_eq(img, 0) 
        rrange = o.max() - o.min() 
        o[ (o - o.min()/rrange) < thresh ] = 0   
        return o  

    def _blue_channel_update(self, img, thresh=1): 
        # if self.color_space == 'lab':
        #     o = img[:,:,-1].copy()
        #     # print( o.min(), o.max() ) 
        #     # 2. now threshold the blue <<< TODO: auto-find the 'unfazzed' pixel <<<< TODO: Is LAB giving more color infor that we can do spectral analysis on or is best jut for pulling out intensity without affecting luminance
        #     o = img_as_ubyte(o)
        #     o[ o != thresh] = 0 #(thresh-o.min()+0.00001)/(o.max() - o.min())] = 0         
        #     o[ o == thresh] = 255  

        # else:
        o = img_as_ubyte(img[:,:,-1].copy() )
        # print( o.min(), o.max() ) 
        o[ o != thresh] = 0 #(thresh-o.min()+0.00001)/(o.max() - o.min())] = 0         
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
    
        # 1. vessels --- using green channel 
        outiez.append( self._vessels_channel(img ) ) 

        # 2. Color INFO
        # 2a. Green Channel clean up @ contrast and full info 
        outiez.append( self._green_channel_update( img )  ) 

        # # 2b. RGB Vs LAB @ red and blue spectrum 
        # if self.color_space == 'lab':
        #     cimg = self._get_lab_img( img )
        # else: 
        #     cimg = img #.copy() 
        # outiez.append( self._red_channel_update( cimg , self.red_thresh) )
        # outiez.append( self._blue_channel_update( img ) )   

        ## For Now Run RGB blue  and LAB red <<< TODO add switch 
        outiez.append( self._red_channel_update(  self._get_lab_img( img ) , self.red_thresh) )
        outiez.append( self._blue_channel_update( img ) )   

        # append yellow for pigmentation
        outiez.append( self._get_yellow_from_rgb2lab(img) )

        # 3.  append origi as cleaned only 
        if self.add_origi:
            _ = [outiez.append(self._get_color_eq(img) ) for i in range(3)]  
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
        c = len(x.shape) 
        if c <= 2:
            o = x
        else:
            o = x[:,:,self.g_channel] #[for c in x] 
        # print( f"From {c} to {o.shape }")
        return o.flatten()  #.reshape(1, -1) #

    def fit(self, X, y=None):
        ## first fit component_selector before transform 
        self.component_selector.fit( [self._get_op_channel(x) for x in X]  )  ## np.vectorize 
        # print("**** FIN FIT ****")
        return self 

    def transform(self, X, y=None):
        ## first fit component_selector before transform 
        return [self.remapped_data(x) for x in X ]   
        
    # per channel 
    def remapped_data(self, img): ## appends to the stack unless otherwise 
        ### TODO: per channel for now operating on one channel only but can append that to the original imag
        x,y, c = img.shape 

        o = self._get_op_channel(img) 

        to = self.component_selector.transform([o,])[0]

        tx =  int( len(to)  * 0.5 * (x/y) )
        to = to.reshape( (tx, -1) )
        o = np.zeros((x,y))
        # print( f"len = {len(to)}, tx = {tx}, to={to.shape} for img={img.shape} on o={o.shape}")
        ox,oy = to.shape 
        o[:ox, :oy] = to 
        
        # print("FIN-Egz: In: ", img.shape, " Out: ", o.shape , " on to=", to.shape) 
        
        if len(img.shape) <= 2:
            return np.dstack([img, o]) if self.append_component else o 
        else:
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

