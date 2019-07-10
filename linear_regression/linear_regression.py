import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from gradientdescent import *

LAMBDA = 0.00001

def ReadExperimentalData():

    m = 100
    x = np.linspace(0, 1, m)
    x.shape = (m, 1)

    y = ( x * 2 ) - 1
    y = y * y

    # Perturb
    z = ( 90 + np.linspace( 0, 360, m ) ) * ( math.pi / 180.0 );
    z.shape = ( m, 1 )
    z = np.sin( z )
    y = y * z

    # Add in higher orders of x
    numHigherOrders = 50
    xOrig = x
    xx = x
    for i in range ( 0, numHigherOrders ):
        xx = xx * xOrig
        x = np.append( x, xx, axis=1 )

    return ( x, y )



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
CheckDimensions( x, y )

# Scale and normalise input
ret = ScaleAndNormalise( x )
x = ret[0]
meanAndSum = ( ret[1], ret[2] )
m = x.shape[0]
n = x.shape[1]

# Split data into training and testing sets
splitData = SplitDatasetIntoTrainingAndTest( x, y, 30 )

x = splitData[0]
y = splitData[1]
CheckDimensions( x, y )
m = x.shape[0]
n = x.shape[1]

xTest = splitData[2]
yTest = splitData[3]
CheckDimensions( xTest, yTest )
assert xTest.shape[1] == n

# Insert the leading column of 1's
x = np.insert( x, 0, 1, axis=1 )
xTest = np.insert( xTest, 0, 1, axis=1 )

# Set up initial theta
theta = np.zeros( (n+1, 1) )

# print( "x=\n", x)
# print( "y=\n", y)

theta = GradientDescent( x, y, theta, LAMBDA, ComputeJ )

# print( numIter, "iterations" )
# print ( "error = ", j )

j = ComputeJ( xTest, theta, yTest, xTest.shape[0] )
print( "result error = ", j)






