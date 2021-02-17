'''
author: bg
goal: 
type: modelz - sklearn for workflow management + keras for transfer learning components + pytorch for nn modules  
how: wrapper class for workflow management (pytorch opt&loss + sklearn pipe&metrics) + ArchitectureMixin and implementation for custom architectures. 
ref: 
refactors: 
'''

from sklearn.base import BaseEstimator 
from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin

## TODO: use same @ transforms 
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import unique_labels 

from torch.nn import nn 
from torch import F 

class ZNNArchitecture(BaseEstimator, nn.Module): ## TODO: VS internal class for nn.Module subclassing  <<< subClassing Vs has-a 
    def __init__(self): 
        ## setup layers and initialize model weights and biases 
        pass 
    ### === PyTorch nn computational graph imple === 
    def forward(self, x): ## x is 1xF 
        return x  
    ### === sklearn pipeline management imple === 
    def fit(self, X, y=None): ## X is NxF, y is Nx1
        return self
    def transform(self, X, y=None):## X is NxF, y is Nx1
        return X
    def score(self, y=None):## y is Nx1
        return -1 


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
    check_estimator( ZNNArchitecture() )  ## check adheres to sklearn interface and standards 
    # TODO: parametrize_with_checks pytest decorator 