import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

def ComputeJ( y, h, l, theta ):
    m = y.shape[0]
    j = -(1/m) * ( np.matmul( y.transpose(), np.log(h) ) + np.matmul( (1-y).transpose(), np.log( 1-h ) ) )
    j = j + ( (1/(2*m)) * ( np.matmul( l.transpose(), theta*theta)) )
    return j

# m=20
# x = np.linspace( -10, 10, num = m )
# x.shape = (m,1)
#
# h = 1 / ( 1 + np.exp(-x))
#
# plt.plot( x, h )
#
# y = np.ones( ( m, 1))
# y[0:(m>>1)] = 0
#
# # print ( y )
#
# j = ComputeJ( y, h)
# print (j )
#
# y[0:(m>>1)] = 1
# y[(m>>1):m] = 0
#
# j = ComputeJ( y, h)
# print (j )
#
# plt.show()

h = np.ones( (1,1) )
y = np.ones( (1,1) )

h[:,:] = 0.9999999999999
# print( -np.log(1-h) )
y[:,:] = 0

# j = ComputeJ( y, h)
# print (j)

l = np.ones( 10 )
l.shape = (10,1)
l[0] = 0

theta = np.linspace( 1, 10, 10 )
theta.shape = (10,1)
# print ( theta * theta)
# print( l.transpose() )
# print (theta)

print ( np.matmul( l.transpose(), theta ))