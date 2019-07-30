
import numpy as np
from slaterwf import ExponentSlaterWF as ewf


class Hamiltonian:
    def __init__(self,Z=2):
        self.Z=Z
    def pot_en(self,pos):
        """ electron-nuclear potential of configurations 'pos' """
        return -np.sum(self.Z/np.linalg.norm(pos, axis=1), axis=0)
    def pot_ee(self,pos):
        """ electron-electron potential of configurations 'pos' """
        return 1/(np.linalg.norm(pos[0]-pos[1], axis=0))
    def pot(self,pos):
        """ potential energy of configuations 'pos' """
        return self.pot_en(pos)+self.pot_ee(pos)


if __name__=="__main__":
    # This part of the code will test your implementation. 
    # Don't modify it!
    pos=np.array([[[0.1],[0.2],[0.3]],[[0.2],[-0.1],[-0.2]]])
    ham=Hamiltonian()
    print("Error:")
    print(ham.pot_en(pos) - -12.0118915)
    print(ham.pot_ee(pos) -  1.69030851)
    print(ham.pot(pos) - -10.321583)
