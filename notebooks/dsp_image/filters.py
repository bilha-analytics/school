import abc
import numpy as np 


class AFilter:
    @staticmethod 
    def pad(img, kern):
        # TODO: add padding remover; return image without padding 
        w, h = img.shape 
        kw, kh = kern.shape
        pw, ph = (kw - 1)//2, (kh-1)//2
        
        pot = np.zeros( (w+pw*2, h+ph*2) )
        pot[ pw:-pw, ph:-ph] = img 
        
        return pot
    
    @staticmethod
    def unpad(img, kern):
        w, h = img.shape 
        kw, kh = kern.shape
        pw, ph = (kw - 1)//2, (kh-1)//2
        
        return img[kw:-kw, kh:-kh]

    @staticmethod 
    def get_kern(k):
        ## flip 180 
        flipped = np.flip(k, 1)
        return flipped
    
    @staticmethod 
    def get_convolve_value(k, phood):
        ## sum of element-wise products
        return np.sum( k * phood)
        
    @abc.abstractmethod
    def apply(img, kern, padit=True):
        NotImplementedError
        
    @staticmethod 
    def convolve(img, kern, padit=True):         
        flipped_kern = AFilter.get_kern(kern)
                
        in_img = AFilter.pad(img, kern) if padit else img.copy() 
        output = np.zeros(in_img.shape)
        
        rowz, colz = in_img.shape 
        kw, kh = kern.shape
        
        ##TODO: fix padding/border-crossing
        for i in range(1, rowz+1):
            for j in range(1, colz+1):
                phood = in_img[ i:i+kw, j:j+kh ]
                output[ i, j] = AFilter.get_convolve_value(flipped_kern, phood)
                
                if (j+kh)>= colz:
                    break
            if (i+kw) >= rowz:
                break
        return output
    
    
class BaseFilter(AFilter):
    @staticmethod
    def apply(img, kern, padit=True):
        outsie = AFilter.convolve(img, kern, padit)
        return AFilter.unpad( outsie, kern) if padit else outsie 
        ## using numpy convolve << BUT is 1-D convolve
        ## return np.convolve(img, kern)
        

class MedianFilter(AFilter):    
    ##TODO: inheritance and static methods
    @staticmethod
    def apply(img, size, padit=True):       
        kern = np.ones((size, size)) 
                
        in_img = AFilter.pad(img, kern) if padit else img.copy() 
        output = np.zeros(in_img.shape)
        
        rowz, colz = in_img.shape 
        kw, kh = kern.shape
        
        ##TODO: fix padding/border-crossing
        for i in range(1, rowz+1):
            for j in range(1, colz+1):
                phood = in_img[ i:i+kw, j:j+kh ]
                output[ i, j] = MedianFilter.get_convolve_value(kern, phood)
                
                if (j+kh)>= colz:
                    break
            if (i+kw) >= rowz:
                break
        return AFilter.unpad(output, kern) if padit else output
    
    ##TODO: inheritance and static methods 
    def get_convolve_value(k, phood):
        return np.median( phood.flatten() )
       
