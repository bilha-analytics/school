import numpy as np 
import skimage
from scipy import ndimage 
from matplotlib import pyplot as plt 

## Prioritize ndimage operations over skimage or the best of the two??
class AnImage:        
    def __init__(self, src, isgray=True):
        self.img = skimage.io.imread( src )
        if isgray:
            self.img = skimage.color.rgb2gray( self.img )
    
    @property
    def rgb(self):
        return skimage.color.gray2rgb(self.img)
    
    @property
    def gray(self):
        return skimage.color.gray2rgb(self.img)
    
    def show(self, cmap='gray', binz=None):
        plt.subplot(1,2,1)
        if (cmap is None):
            plt.imshow(self.img)
        else:
            plt.imshow(self.img, cmap=cmap) 
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.hist(self.img.flatten()*(1/self.img.max()), bins=binz)
        plt.title(f"Histogram: {binz} bins")
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        
    def contour(self):
        plt.contour(self.img) ##TODO: more at levels 
        
    @property
    def noisy(self):
        outi = self.img.copy()
        outi = outi + 1.4 * outi.std() * np.random.random(outi.shape)
        return outi
    
    @property
    def stats(self): 
        return f"---Image Stats---\n \
Shape: {self.img.shape} \n \
Type: { type( self.img )} \n \
Mean: { np.mean( self.img )} \n \
Median: {np.median( self.img )} \n \
Max: {np.max( self.img )} \n \
Min: {np.min( self.img )} "
    
    def denoised(self, sig=3):
        # Gaus will smooth but at expense of edges so keep sig small
        return ndimage.gaussian_filter(self.img, sig)
    
    def colorised(self, r=100, g=20, b=0):
        ## Luma transform weights 
        # LR = wR + xG + yB
        # (w,x,y) = (299, 587, 114)/1000
#         colored =  skimage.img_as_float(self.make_3d() )
#         Ltrans = np.array([299, 587, 114])*(1/1000)
#         R, G, B = colored[:,:,0], colored[:,:,1], colored[:,:,2]
#         colored = (R*700 + G*587 + B*114)/1000

        colored = self.make_3d()
        colored[:,:,0] = r
#         colored[:,:,1] = g
#         colored[:,:,2] = b
        return colored
            
    
    def make_3d(self):
        if len(self.img.shape) >= 3:
            return self.img.copy()
        
        nr, nc = self.img.shape
        f = np.empty( (nr, nc, 3))
        ##brute force
        for i in range(nr):
            for j in range(nc):
                for k in range(3):
                    f[i,j,k] = img.img[i,j]*(i*j)
        return f
    
class ArrayImage(AnImage):    
    def __init__(self, matrix):
        self.img = matrix