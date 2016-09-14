import numpy as np
import pandas as pd

class CompositionalDataException(Exception):
    '''This exception is raised when an object is not of CompositionalData class.'''
    def __init__(self,arguments):
        print("{arg} is not of CompositionalData class. Set it as CompositionalData first.".format(arg=self))
    
class CompositionalData:
    """Class definition of compositional data.
    
    Compositional data are positive data with a constant sum. 
    Their statistical analysis require special consideration as, due to the 
    constant sum, compositional data define an (n-1) dimensional space where 
    n is the number of components."""
    
    def __init__(self,__values=None,index_name=None,column_name=None):
        """Initiate compositional data class."""
        
        #If index and column names are not given, name them as p1,..,,pn and c1,...,cn, respectively.
        if __values is None:
            self.__values = pd.DataFrame(np.array([]))
        else:
            self.__values = pd.DataFrame(np.array(__values))
        if index_name is None:
            self.__values.index = ["c{num}".format(num=number) for number in range(1,self.__values.shape[0]+1)]
        if column_name is None:
            self.__values.columns = ["p{num}".format(num=number) for number in range(1,self.__values.shape[1]+1)]
        self.total = np.sum(self.__values,axis=1)
     
    def close(self):
        """Closure operation rescales the compositions to have a unit sum."""
        return self.__values.div(self.total,axis=0) # divide row-wise
    
    def power(self,power):
        """Powering raises the composition to power elementwise and then closes it."""
        powered = self.__values.pow(other=power)
        return powered.div(np.sum(powered,axis=1),axis=0)
    
    def perturb(self,other_composition):
        """Perturbation multiplies the composition with an other composition elementwise and closes it."""
        if type(other_composition) is not CompositionalData:
            raise CompositionalDataException(other_composition)
        multiplied = self.__values.mul(other_composition.get_values())
        return multiplied.div(np.sum(multiplied,axis=1),axis=0)
    
    def __getitem__(self,indices):
        """Returns entries indexed by a 2-tuple using standard Python notation (Pandas DataFrame.ix method)"""
        row,column = indices
        return self.__values.ix[row,column]
    
    def get_values(self):
        """Returns the composition(s) as a Pandas data frame."""
        return self.__values
    
class CompositionalTransform:
    """Compositional transform class which operates on compositional data.
    
    The standard Euclidean n-dimensional basis does not provide an orthonormal basis in the compositional space.
    Because of this, the proper statistical space of compositional data is the (n-1) simplex. However, 
    compositional transformations map the composition from the (n-1) simplex to the Euclidean space allowing the 
    use of standard multivariate statistical techniques. The dimension of the Euclidean space depends on the type
    of the transform. 
    
    The additive log-ratio transforms the (n-1) simplex into the (n-1) dimensional Euclidean space. Note that it does not
    provide an orthonormal basis."""
    
    # additive log-ratio transform
    def alr(compositional_data,divisor=None):
        """Returns the alr transformed coordinates by leaving divisor out and dividing by it."""
        if type(compositional_data) is not CompositionalData:
            raise CompositionalDataException(compositional_data)
        if divisor is None:
            divisor = compositional_data.get_values().columns[-1] 
        divisor_part = compositional_data.get_values().loc[:,divisor]
        return np.log(compositional_data.get_values().drop(divisor,axis=1).div(divisor_part,axis=0))

class InverseCompositionalTransform:
    """Inverse compositional transform class which transforms the real coordinates back to the simplex."""
    
    def inv_alr(transformed_data,column_names=None):
        """Returns the simplicial values from the alr coordinates."""
        # if no column_names are given, assume the last column was involved in the alr-transform and name it p#last_column
        if column_names is None:
            index = len(transformed_data.columns)
            column_name = "p"+str(index+1)
        else:    
            # find the parts that were included in the transform
            in_alr = [(index,which_column) for index,which_column in enumerate(column_names)
                      if which_column not in transformed_data.columns]
            index,column_name = in_alr[0]
        # insert zeros for the column that was left out of the alr transform and exponentiate 
        transformed_data.insert(index,column_name,np.zeros(len(transformed_data.index)))
        return CompositionalData(np.exp(transformed_data)).close()
      
