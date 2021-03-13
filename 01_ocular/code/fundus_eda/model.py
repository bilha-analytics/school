'''
author: bg
goal: workflow and architecture 
type: fundus EDA specific setup
how: use mflow as base 
ref: 
refactors: abstract and move common components to mflow 
'''

from mflow import utilz, report 
from mflow import model, nnarchs, learning_bits 
from mflow import zdata, preprocess, extract

from sklearn.pipeline import Pipeline

class NNConvAttentionMulti(nnarchs.ZNNArchitecture):

    def __init__(self, n_classes):
        super().__init__(n_classes=n_classes) 


