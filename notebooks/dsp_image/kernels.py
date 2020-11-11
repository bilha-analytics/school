'''
author: bg
usage: 
Utils module. 
Collection of functions for creating different types of kernels  
'''
import numpy as np

def make_gaussian_kern(size, sig=1):    
    ## size better if odd b/c balance
    ## The larger the size of the kernel, more blur <<< the lower the sensitivity to noise << 
    outk = np.zeros( (size, size) )
    
    k = (size-1)//2
    sig_const = 1/( 2 * np.pi * sig) 
    
    for i in range(size):
        for j in range(size):
            outk[i,j] = sig_const * np.exp( ( ( (i+1)-(k+1))**2 +( (j+1)-(k+1))**2 )*(-1/(2*sig)))
    return outk


def make_box_kern(size, k=None): 
    ## average or box blur
    outk = np.ones( (size, size) )
    kn = outk.sum() if k is None else k 
    outk = outk * (1/kn) 
    return outk


def make_laplacian_ker(size):
    ##TODO: more options 
    return np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

def make_edge_soft1(size):
    ##TODO: more
    return np.array([
                [1, 0, -1],
                [0, 0, 0],
                [-1, 0, 1]
            ])    
def make_edge_harsh1(size):
    ##TODO: more
    return  np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ])  


