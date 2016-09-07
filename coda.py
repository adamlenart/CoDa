import numpy as np

class CompositionalData:
    """Compositional data are positive data with a constant sum. 
    Their statistical analysis require special consideration as, due to the 
    constant sum, compositional data define an (n-1) dimensional space where 
    n is the number of components. Therefore, the standard Euclidean n-dimensional
    basis does not provide an orthonormal basis in the compositional space.
    Because of this, the proper statistical space of compositional data is the 
    (n-1) simplex."""
    
    def __init__(self,values=None):
        if values is None:
            self.values = np.array([])
        else:
            self.values = np.array(values)
        self.total = sum(self.values)
    def close(self):
        return self.values/self.total