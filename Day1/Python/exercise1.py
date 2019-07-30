import numpy as np
import scipy as sp
import scipy.special

N=20
elem = 200
# fill in code here
# when x should be a random number between 1 and 2
# estimates should hold the first 10 partial sums of the taylor series of exp(x)
x = np.random.rand(elem)+1
estimates = 0.
for i in range(N):
    fact = sp.special.factorial(i)
    power = np.power(x, i)
    estimates += power/fact

print("x=",np.mean(x), "Exp(x)=",np.mean(np.exp(x)))
print(np.mean(estimates))
print(np.mean(estimates-np.exp(x)))
