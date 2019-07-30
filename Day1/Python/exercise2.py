import numpy as np

npts=int(1e8)
pos=np.random.randn(npts,3)
#pos holds npts points in 3 dimensions where x,y and z are chosen
#as normally distributed around zero

#want to find the distance form the origin for each point in a new array dist
dist = np.linalg.norm(pos, axis=1)

#print(pos)
#print(dist)
print("Mean dist: {}\tDist expectation: {}".format(np.mean(dist), np.linalg.norm([0.5, 0.5, 0.5])))

