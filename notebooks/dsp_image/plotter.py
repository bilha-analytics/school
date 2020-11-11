
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt 


import skimage
from scipy import ndimage

from image import AnImage



def show_image_list(img_list, nc=2, cmap='gray', titlez=None):    
    n = len(img_list)
    nr = n//nc + ( 0 if n%nc == 0 else 1) 
    for i, img in enumerate(img_list):
        plt.subplot(nr, nc, (i+1) )
        plt.imshow( img, cmap=cmap)
        plt.axis('off')
        if titlez:
            plt.title( f"{titlez[min(i, len(titlez)-1)]}" )
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show();


class Junkie:
    @staticmethod 
    def equalize_histogram(img, binz=20, method=1):
        p2, p98 = np.percentile(img, (2,98))
        method_str = ['rescale', 'eq_hist', 'adapt_hist', '']
        method_fx = [
            (skimage.exposure.rescale_intensity, {'in_range':(p2,p98)}),
            (skimage.exposure.equalize_hist, {}),
            (skimage.exposure.equalize_adapthist, {'clip_limit':0.03}),
        ]
        
        plt.title(f"{method_str[method]}")
        plt.subplot(2,2,1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(2,2,2)
        plt.hist(img.flatten(), bins=binz, color='b')
        cdf, cbinz = skimage.exposure.cumulative_distribution(img, binz)
        plt.plot(cbinz, cdf, 'r')
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

        fx = method_fx[method]
        img_eq = fx[0](img, **fx[1])

        plt.subplot(2,2,3)
        plt.imshow(img_eq, cmap='gray')
        plt.axis('off')

        plt.subplot(2,2,4)
        plt.hist(img_eq.flatten(), bins=binz, color='b')
        cdf, cbinz = skimage.exposure.cumulative_distribution(img_eq, binz)
        plt.plot(cbinz, cdf, 'r')
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

        return img_eq
    
    @staticmethod
    def equlization_flow(fpath, nc=2):
        img = AnImage(fpath)

        print('Default Equalize')
        eq1 = Junkie.equalize_histogram(img.img, method=1)
        plt.show()
        plt.clf();

        print('Adaptive Equalize')
        eq2 = Junkie.equalize_histogram(img.img, method=2)
        plt.show()
        plt.clf();

        print('Percentile Rescaling - Contrast Stretching')
        eq0 = Junkie.equalize_histogram(img.img, method=0)
        plt.show()
        plt.clf();
        
        show_image_list([img.img, eq1, eq2, eq0], titlez=['origi', 'default equi', 'adapt-hist', 'percentile'], nc=nc)
        plt.show(); 