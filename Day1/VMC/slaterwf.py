import numpy as np


class ExponentSlaterWF:
    """ 
    Slater determinant specialized to one up and one down electron, each with
    exponential orbitals.
    Member variables:
        alpha: decay parameter.

    Note:
    pos is an array such that
        pos[i][j][k]
    will return the j-th component of the i-th electron for the 
    k-th sample (or "walker").
    """
    def __init__(self,alpha=1):
        self.alpha=alpha
#-------------------------
    
    def value(self,pos):
        r = np.linalg.norm(pos, axis=1)
        return np.exp(-r[0]*self.alpha)*np.exp(-r[1]*self.alpha)
#        pass # Implement me!
#        return np.zeros(pos.shape[2])
#-------------------------
    def gradient(self,pos):
        prefac = -self.alpha/np.linalg.norm(pos, axis=1)
        return np.multiply(pos, prefac.reshape(pos.shape[0],1,pos.shape[2]))
#        pass # Implement me!
#        return np.zeros(pos.shape)
#-------------------------
    def laplacian(self,pos):
        return -(2*self.alpha)/np.linalg.norm(pos, axis=1) + self.alpha**2
#        pass # Implement me!
#        return np.zeros((pos.shape[0],pos.shape[2]))
#-------------------------


if __name__=="__main__":
    # This part of the code will test your implementation. 
    # Don't modify it!
    import wavefunction
    # 2 electrons, 3 dimensions, 5 configurations.
    testpos=np.random.randn(2,3,5)
    print("Exponent wavefunction")
    ewf=ExponentSlaterWF(0.5)
    wavefunction.test_wavefunction(ExponentSlaterWF(alpha=0.5))
    
