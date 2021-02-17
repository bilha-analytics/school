'''
author: bg
goal: 
type: modelz - sklearn for workflow management + PyTorch/keras for transfer learning components + pytorch for nn modules  
how: wrapper class for workflow management (pytorch opt&loss + sklearn pipe&metrics) + ArchitectureMixin and implementation for custom architectures. 
ref: 
refactors: 
'''

from sklearn.base import BaseEstimator 
from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.pipeline import Pipeline 

## TODO: use same @ transforms 
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import unique_labels 

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nnarchs 
from zdata import ZSerializableMixin 

class ZModel(ZSerializableMixin, BaseEstimator): ## TODO: VS internal class for nn.Module subclassing  <<< subClassing Vs has-a 
    ## cuda, optimizer, loss, evaluate 
    
    MSG_LOSS_PER_EPOCH = "Epoch {:5d}: Loss: {:15.4f} \tn = {:5d}".format
    MSG_ACC_ON_PREDICT = "Predict n ={:5d}: Acc: {:15.4f}".format
    MSG_YHAT_Y_VALZ = "{:3d}. {:4.2f} ===> {:4.2f}".format

    def __init__(self, nnModel=None,  ##TODO: non-nn Models and traditional ML 
                use_cuda=False,  ##TODO: pass to ZNNArch 
                epochs=3,
                loss_func=(nn.CrossEntropyLoss, {}) ,  
                optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9}) ): 
        ## setup layers and initialize model weights and biases 
        self.nnModel = nnModel
        self.epochs = epochs 
        self.loss_func = loss_func 
        self.optimizer = optimizer
        self.loss_thresh = 1e-6
        ## TODO: cuda calls  
        self.use_cuda = use_cuda 
        self.init_cuda()
    
    ### === sklearn pipeline management imple === 
    def fit(self, X, y=None): ## X is NxF, y is Nx1
        ## train model + batches 
        self.train(X, y) 
        return self
    def transform(self, X, y=None):## X is NxF, y is Nx1
        ## predict with no_grad 
        return self.predict(X, y, log=False)

    def score(self, yhat, y):## y is Nx1
        ACC_ = np.array( [ int( int(y) == int(yh) ) for y, yh in zip(y, yhat)]).mean()  
        print( self.MSG_ACC_ON_PREDICT( len(y), ACC_ ) )  
        return ACC_

    def init_cuda(self):
        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print('Cuda is not available')
        self.device = torch.device('cpu')
    
    ## TODO: returns + skl.metrics
    def train(self, X_train, y_train=None):
        N_ = len(X_train) 
        lossor = self.loss_func[0]( **self.loss_func[1])
        optimor = self.optimizer[0]( self.nnModel.parameters(), **self.optimizer[1])
        def fwd_pass(x, y=None, ebatch=0): 
            ## 1. train on batch 
            self.nnModel.train() ## set to training mode 
            outz = self.nnModel.forward( x )
            l_ = lossor( outz, y)   

            optimor.zero_grad()
            l_.backward() 
            optimor.step()  
            
            ## 2. evaluate batch ##TODO: ensure batches
            if ebatch % 10 == 0 or ebatch == self.epochs:
                self.nnModel.eval() 
                outz = self.nnModel.forward(x) 
                l_ = lossor( outz, y) 
            return l_.item()

        LOSS_ = 0.
        for e in range( self.epochs):
            if isinstance(X_train, DataLoader):
                pass
            else:
                if y_train is None:
                    for x in X_train: 
                        LOSS_ += fwd_pass(x, ebatch=e+1) 
                else:
                    for x, y in zip(X_train, y_train): 
                        LOSS_ += fwd_pass(x, y, e+1)
            print( self.MSG_LOSS_PER_EPOCH( e+1, LOSS_ ,N_ ) )  
            ## TODO: better flow and when to stop 
            if LOSS_ <= self.loss_thresh:
                break 

    ## TODO: returns + skl.metrics
    def predict(self, X_, y_, log=True): 
        def eval_pass(x):
            with torch.no_grad():
                self.nnModel.eval()
                outz = self.nnModel.forward(x) 
                return torch.argmax( outz  ).cpu().numpy() 

        yhat = []
        if isinstance(X_, DataLoader):
            pass
        else:
            for x in X_: 
                yhat.append( eval_pass(x) )
        
        if log:        
            ACC_ = self.score(yhat, y_) 
            print( self.MSG_ACC_ON_PREDICT( len(y_), ACC_ ) )  
            for i, (y, yh) in enumerate(zip(y_, yhat)):
                # print(y, yh)
                print( self.MSG_YHAT_Y_VALZ(i+1, int(y), yh))
        return yhat 


### --- TBD: Skorch == sklearn + pytorch already! for now i want to moi-flow 
### --- TODO: Captum == Model intepretability for PyTorch 
class ZModelManager():
    ### === workflow management, permutations and hyperparam tuning configuration 
    def __init__(self): ## NO:ByNNs = setup layers and initialize model weights and biases 
        # workflow management, permutations and hyperparam tuning configuration 
        pass 
    ### === metrics and evaluation ===

    ### === 





if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator 
    # check_estimator( ZModel() )  ## check adheres to sklearn interface and standards 
    # TODO: parametrize_with_checks pytest decorator  
    epochz = 3 
    N, nf, nclasses  =  12, 40, 2 #.reshape(1, -1)
    tmpX = [ torch.tensor( np.random.randint(0, 100, size=nf).reshape(1, -1).astype(np.float32) ) for i in range(N)]
    tmpY = [  x.sum()**2 for x in tmpX]
    ymu = np.array(tmpY).mean() 
    tmpY = [ torch.tensor( np.array([ int(y > ymu),] ).astype(np.long) ) for y in tmpY] ## TODO: qcut percentile


    tmpX = tmpX * 300
    tmpY = tmpY * 300
    n_train = int(len(tmpY)*0.75) 

    mlp = nnarchs.ZNNArchitectureFactory.mlp(nf, nclasses) 
    print(mlp )
    
    
    model = ZModel( mlp, epochs=epochz, 
                    loss_func=(nn.CrossEntropyLoss, {} ), 
                    optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9} ) )
    print( model )
 
    model.train(tmpX[:n_train], tmpY[:n_train]) 

    yhat = model.predict(tmpX[n_train:], tmpY[n_train:] , log=False)
    model.score(yhat, tmpY[n_train:])

    c = "="
    print(f"{c*10} End Model --> Dumping to File {c*10}\n")

    fpath = "./tester.zmd"
    model.dump(fpath )
    model2 = ZModel() 
    model2.load(fpath)
    model2.predict(tmpX, tmpY , log=False)

    model2.epochs =5
    model2.fit(tmpX[:n_train], tmpY[:n_train])
    yhat = model2.transform(tmpX[n_train:], tmpY[n_train:])
    model2.score(yhat, tmpY[n_train:])

    print(f"{c*10} End Model2 <-- Loaded from file, predicted, retrained and predicted {c*10}\n")

    epochz = 5
    # print( type(tmpX), type(tmpY))
    # print( len(tmpX), len(tmpY))
    tmpX = tmpX * 3
    tmpY = tmpY * 3
    n_train = int(len(tmpY)*0.75) 
    # print( len(tmpX), len(tmpY))
    mlp = nnarchs.ZNNArchitectureFactory.mlp(nf, nclasses, {'n_layers':3} ) 
    model3 = ZModel( mlp, epochs=epochz, 
                    loss_func=(nn.HingeEmbeddingLoss, {} ), 
                    optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9} ) )
    # print( model3 )
    model3.fit( tmpX[:n_train], tmpY[:n_train])
    yhat = model3.transform(tmpX[n_train:], tmpY[n_train:])
    model3.score(yhat, tmpY[n_train:])

    print(f"{c*10} End Model3 <-- new, hyperparams {c*10}\n")
    # ## TODO: classifier/regressor/clusterer/etc Mixin requirements
    # piper = Pipeline(['model', model2])
    # print( piper )

    # piper.fit_transform(tmpX, tmpY)