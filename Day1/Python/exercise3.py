import numpy as np; import scipy as sp
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt

nsamples=18 # number of sample sizes
estimate=np.zeros(nsamples)*1.0 # data arrays
error=np.zeros(nsamples)*1.0


#for j in np.arange(nsamples)+1:
#    npts=np.power(2,j) # double sampling points each time
#    acircle=np.zeros(npts)
#    #generate points inside the unit square
#    # fill this in
#    pts = np.random.rand(npts, 2)
#    inside = 0
#    outside = 0
#    for i in range(len(pts)):
#        # check if points are inside the unit circle an if so record in acircle
#        # fill this in
#        dist = np.sqrt(np.sum(np.square(pts[i])))
#        if dist <= 1.0:
#            inside += 1
#            acircle[i] = 1
#    #put the average number of points falling in the circle * 4 in estimate[j-1]
#    #the error can be estimated using sp.stats.sem()
#    #fill this in
#    area = 1 - np.mean(acircle)
#    estimate[j-1] = 4*(inside/npts)
#    error[j-1] = 4*sp.stats.sem(acircle)
#    print (estimate[j-1], "+/-", error[j-1])
#print(np.pi)
    
