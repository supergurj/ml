import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import math

def ReadHousingData():
    # Load csv file
    df = pd.read_csv("house_data_small.csv")

    x = df[["sq_feet", "num_bedrooms", "num_bathrooms"]].values
    y = df[["sale_price"]].values

    return ( x, y )

def ReadExperimentalData():

    x = np.linspace(0, 10, 100)
    x.shape = (100, 1)

    y = x * x * x

    # Perturb
    z = np.linspace( 0, 720, 100 ) * ( math.pi / 180.0 )
    z = np.sin( z ) * 100.0
    z.shape = ( 100, 1 )

    y = y + z

    x = np.append( x, x * x, axis=1 )

    return ( x, y )

def ScaleAndNormalise( v ):

    mean = np.mean(v, axis=0)
    sum = np.sum(v, axis=0)

    v = np.subtract(v, mean)
    v = np.divide(v, sum)

    return (v, mean, sum )

def ComputeJ( x, theta, y, m ):

    err = np.matmul( x, theta) - y
    err.shape = ( m, )  # Convert to 1D array
    j = np.dot( err, err ) * (0.5/m)
    return j


# Read input as tuple of np arrays
# input = ReadHousingData()
input = ReadExperimentalData()

# Set up x and y
x = input[0]
y = input[1]

# Check dimensions
assert len( x.shape ) == 2
assert x.shape[0] == y.shape[0]

m = x.shape[0]
n = x.shape[1]

# Scale and normalise input
ret = ScaleAndNormalise( x )
x = ret[0]
meanAndSum = ( ret[1], ret[2] )

# Insert the leading column of 1's
x = np.insert( x, 0, 1, axis=1 )

# Set up initial theta
theta = np.zeros( (n+1, 1) )

# print( "x=\n", x)
# print( "y=\n", y)

# Gradient descent

ALPHA_START = 10.0
ALPHA_SCALE = 0.5
ALPHA_MIN = 0.00001
TERM_MAX_ITERATIONS = 10000000
TERM_MIN_ERROR_RELATIVE_DELTA = 0.0000001

j = ComputeJ( x, theta, y, m )
alpha = ALPHA_START

numIter = 0
while 1:

    w = np.matmul( x.transpose(), np.matmul( x, theta) - y )
    thetaNew = theta - (alpha/m) * w
    jNew = ComputeJ( x, thetaNew, y, m )

    if ( jNew < j ):

        # Update theta
        theta = thetaNew

        # Check for termination
        if ( (j - jNew) / j ) < TERM_MIN_ERROR_RELATIVE_DELTA :
            print( "Terminating after relative error decreasing below threshold.\n")
            break

        if ( numIter > TERM_MAX_ITERATIONS ) :
            print( "Terminating after %d iterations.\n" % numIter )
            break

        j = jNew
        numIter = numIter + 1

    else:

        if ( alpha < ALPHA_MIN ):
            # Time to quit
            print( "Terminating after relative error started increasing with min alpha.\n")
            break
        else:
            # Try smaller step
            alpha = alpha * 0.5



result = np.matmul( x, theta )

plt.figure()
xaxis = x[:, 1]

# print("theta = ", theta)
# print ( "result =", result )
# print( xaxis )

plt.plot( xaxis, y, label="y" )
plt.plot( xaxis, result, label = "result" )
plt.legend()
plt.show()




