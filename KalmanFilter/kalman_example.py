# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:13:22 2018

@author: plane
"""

# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw
#
# Vectorization and sloping trend line modifications
# by Peter Lane

import numpy
import numpy.linalg as linalg
import pylab

# intial parameters
n_iter = 50
sz = (n_iter, 2, 1) # size of array
x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
slope = numpy.reshape(numpy.arange(0, n_iter, 1), (n_iter, 1))
z = slope+numpy.random.normal(x,8,size=(n_iter, 1)) # observations (normal about x, sigma=0.1)

#Q = 1e-5 # process variance
Q = 0.0

# allocate space for arrays
xhat=numpy.zeros(sz) # a posteri estimate of x
P=numpy.zeros((n_iter, 2, 2))         # a posteri error estimate
xhatminus=numpy.zeros(sz) # a priori estimate of x
Pminus=numpy.zeros((n_iter, 2, 2))    # a priori error estimate
K=numpy.zeros((n_iter, 2, 1))         # gain or blending factor

F = numpy.array([[1, 1], [0, 1]])

H = numpy.array([[1, 0]])
R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = numpy.array([[0.0], [0.0]])
P[0] = numpy.eye(2)

for k in range(1,n_iter):
    # time update
    xhatminus[k] = numpy.dot(F, xhat[k-1])
    Pminus[k] = numpy.dot(F, numpy.dot(P[k-1], F.transpose()))+Q

    # measurement update
    # K[k] = Pminus[k]/( Pminus[k]+R )
    K[k] = numpy.dot(numpy.dot(Pminus[k], H.transpose()), linalg.inv(numpy.dot(H, numpy.dot(Pminus[k], H.transpose())) + R))
    # xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    xhat[k] = xhatminus[k]+numpy.dot(K[k],(z[k]-numpy.dot(H, xhatminus[k])))
    # P[k] = (1-K[k])*Pminus[k]
    P[k] = Pminus[k] - numpy.dot(K[k], numpy.dot(H, Pminus[k]))

print('Velocities: {}'.format(xhat[:,1,0]))

pylab.figure()
pylab.plot(z,'bo-',label='noisy measurements', markerfacecolor='None')
pylab.plot(xhat[:,0,0],'mo-',label='a posteri estimate', markerfacecolor='None')
pylab.plot(x+slope[:,0],color='g',label='truth value')
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage')

pylab.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
pylab.plot(valid_iter,Pminus[valid_iter, 0, 0],label='a priori error estimate')
pylab.xlabel('Iteration')
pylab.ylabel('$(Voltage)^2$')
pylab.setp(pylab.gca(),'ylim',[0,.01])

pylab.figure()
pylab.plot(xhat[:,1,0], label='velocities')
pylab.axhline(1.0, color='g', label='true velocity')
pylab.xlabel('Iteration')
pylab.ylabel('dV/dt')
pylab.show()
