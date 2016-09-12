import numpy as np
import pandas as pd

class CompositionalData:
    """Compositional data are positive data with a constant sum. 
    Their statistical analysis require special consideration as, due to the 
    constant sum, compositional data define an (n-1) dimensional space where 
    n is the number of components. """
    
    def __init__(self,values=None,index_name=None,column_name=None):
        if values is None:
            self.values = pd.DataFrame(np.array([]))
        else:
            self.values = pd.DataFrame(np.array(values))
        if index_name is None:
            self.values.index = ["c{num}".format(num=number) for number in range(1,self.values.shape[0]+1)]
        if column_name is None:
            self.values.columns = ["p{num}".format(num=number) for number in range(1,self.values.shape[1]+1)]
        self.total = np.sum(self.values,axis=1)
    #closure operation    
    def close(self):
        # divide rowwise
        return self.values.div(self.total,axis=0)

    
class CompositionalTransform:
    
    """The standard Euclidean n-dimensional basis does not provide an orthonormal basis in the compositional space.
    Because of this, the proper statistical space of compositional data is the (n-1) simplex. However, 
    compositional transformations map the composition from the (n-1) simplex to the (n-1) Euclidean space allowing the 
    use of standard multivariate statistical techniques. Only alr so far."""
    
    # additive log-ratio transform
    def alr(compositional_data,rm_index=None):
        if rm_index is None:
            rm_index = compositional_data.values.columns[-1] 
        rm_part = compositional_data.values.loc[:,rm_index]
        return np.log(compositional_data.values.drop(rm_index,axis=1).div(rm_part,axis=0))
