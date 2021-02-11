'''
author: bg
goal: 
type: util 
how: 
ref: 
refactors: 
'''

import os, time, glob #TODO: pathlib 
import pickle 

import numpy as np 

import skimage 
from skimage import io, color , img_as_float, exposure, transform, filters 

import matplotlib.pyplot as plt 

## === Image Utilz
class Image:

    CLR_HSV = 0
    CLR_YUV = 1
    CLR_CIE = 2
    CLR_XYZ = 3 
    COLOR_SPACES = [color.rgb2hsv,
                   color.rgb2yuv,
                   color.rgb2rgbcie,
                   color.rgb2xyz,
                   ]
    @staticmethod
    def fetch_and_resize_image(fpath, size):
        try:
            return Image.resize_image_dim( io.imread( fpath ), dim=size) 
        except:
            return None 

    @staticmethod
    def get_channel(img, cid): 
        c = len(img.shape)
        nc = img.shape[2] 
        return img[:,:,cid] if (c >=3 and cid<nc) else None 

    @staticmethod 
    def get_colorspace(img, cid):
        return Image.COLOR_SPACES[cid]( img )

    @staticmethod
    def rescale_and_float(img):
        o = img.copy()
        o = o/255 ## assumes 
        o = img_as_float(o) ## recheck with o/255  
        return o 

    @staticmethod
    def resize_image_perc(img, p=0.25):
        o = transform.rescale(img, p, anti_aliasing=True, multichannel=True)   
        return o
    
    @staticmethod
    def resize_image_dim(img, dim=(50,50) ): ##TODO: aspect ratio and padding to max dim
        return transform.resize(img, dim, anti_aliasing=True )
    
    @staticmethod
    def hist_eq(img, mtype=1): ## TODO: mtype consts
        p2, p98 = np.percentile(img, (2,98)) 
        mtypez = [
            (exposure.equalize_adapthist, {'clip_limit':0.03}),
            (exposure.rescale_intensity, {'in_range':(p2,p98)}), 
        ]
        pmod, kargz = mtypez[mtype]
        return pmod(img, **kargz)
   
    @staticmethod
    def edgez(img, mtype=0): ## TODO: mytpe  
        sharez = {} #TODO: shared params @ API setup 
        mtypez = [
            (filters.frangi, {'sigmas':[0,0.5,1], 'black_ridges':1}), 
        ]
        pmod, kargz = mtypez[mtype] 
        return pmod(img, **kargz)

    @staticmethod
    def denoise(img, mtype=0): ##TODO
        o = img.copy() 
        return o
     
    @staticmethod #TODO: histo eq non-gray imagez
    def basic_preproc_img(img, dim=(50,50), denoise_mtype=0): #image resize, rescale, equalize as bare minimum preprocessing
        return Image.resize_image_dim(
                Image.rescale_and_float(img), dim
            )
        # return img 
    @staticmethod
    def plot_images_list(img_list, titlez=None, nc=2, cmap=None, tstamp=False, spacer=0.01, 
                         save=None , tdir=".", savedpi=800, withist=False, binz=None):
       
        if withist:   
            n = len(img_list)*2
            nr = n//nc + ( 0 if n%nc == 0 else 1) 
        else:
            n = len(img_list)
            nr = n//nc + ( 0 if n%nc == 0 else 1) 
            
        ## image rows
        for i, img in enumerate(img_list):
            plt.subplot(nr, nc, (i+1) )
            plt.imshow( img, cmap=cmap)
            plt.axis('off')
            if titlez and (i<len(titlez)):
                plt.title( f"{titlez[i]}" ) #min(i, len(titlez)-1)
        
        ## histo rows 
        if withist:      
            for i, img in enumerate(img_list):
                plt.subplot(nr, nc, (i+1)+(n//2) )
                plt.hist(img.flatten()*(1/img.max()), bins=binz)
                plt.tick_params(axis='y', which='both', labelleft=False, labelright=False) #TODO:off
                
        plt.subplots_adjust(wspace=spacer, hspace=spacer)
        
        if save:
            d = datetime.now().strftime("%H%M%S")
            fnout = f"{d}_{save}" if tstamp else f"{save}"
            plt.savefig(f"{tdir}/{fnout}.png", dpi=savedpi)
        
        plt.show();
    
### ==== FileIO 
class FileIO:
    @staticmethod
    def file_content(fpath, has_header_row=False, rec_parser=None, sep='\t'):
        with open(fpath, 'r') as fd:
            # print( fpath )
            for rec in fd.readlines(): 
                # print(rec)
                yield rec if rec_parser is None else rec_parser(rec, sep) 
    
    @staticmethod
    def folder_content(fpath, ext="*.*", additional_info_func=None, fname_parser=None, sep='-'): 
        for f in sorted(glob.glob(f"{fpath}/{ext}")): 
            fname = os.path.splitext( os.path.basename(f) )[0]
            fname = [fname,] if fname_parser is None else fname_parser(fname, sep)
            # print( fname )
            xtraz = [] 
            if additional_info_func:
                xtraz = additional_info_func(f) 
            yield [*fname, *xtraz] 
    
    @staticmethod
    def row_parser(rec, sep='\t'):
        outiez = rec.strip().split(sep)
        return [x.strip() for x in outiez if len(x) > 0] ##TODO: clean up paranoia 
    
    @staticmethod
    def image_file_parser(fpath):
        # print( fpath )
        outiez = []
        img = io.imread(fpath)
        outiez.append( img.shape ) 
        outiez.append( img.min() ) 
        outiez.append( img.max() ) 
        outiez.append( img.mean() ) 
        outiez.append( img.std() )
        img = None 
        return outiez 
