import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from gradientdescent import *

LAMBDA = 0.00001

def ReadHousingData():
    # Load csv file
    df = pd.read_csv("house_data_small.csv")

    x = df[["sq_feet", "num_bedrooms", "num_bathrooms"]].values
    y = df[["sale_price"]].values

    return ( x, y )

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

def CheckDimensions( x, y ):
    assert len( x.shape ) == 2
    assert len( y.shape ) == 2
    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == 1
    return

def SplitDatasetIntoTrainingAndTest( x, y, testPercentage ):

    # work out how many samples to use for test data
    m = x.shape[0]
    n = x.shape[1]
    numTestSamples = int( m * ( testPercentage / 100.0 ) )

    print( "m = %d, num test samples = %d\n" %(m, numTestSamples ) )

    xTraining = x
    yTraining = y

    xTest = np.empty( (numTestSamples, n) )
    yTest = np.empty( (numTestSamples, 1) )

    # randomly extract rows as test samples
    for i in range (0, numTestSamples):
        row = np.random.randint( 0, m )

        print ( row )

        xTest[i:i+1:,] = xTraining[row:row+1:,]
        yTest[i:i+1:,] = yTraining[row:row+1:,]

        xTraining = np.delete( xTraining, row, 0 )
        yTraining = np.delete( yTraining, row, 0 )

        m = xTraining.shape[0]

    return ( xTraining, yTraining, xTest, yTest )

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

fig, axs = plt.subplots( 4, 1)

xaxis = x[:, 1]
result = np.matmul( x, theta )
axs[0].plot( xaxis, y, "b.", label="y" )
axs[0].plot( xaxis, result, "orange",  label = "result" )

sortIdx = xTest[:,1].argsort()
xTest = xTest[ sortIdx ]
yTest = yTest[ sortIdx ]
xaxis = xTest[:, 1]

result = np.matmul( xTest, theta )
axs[1].plot( xaxis, yTest, "g.", label="test")
axs[1].plot( xaxis, result, "orange",  label = "result" )


# if numTraces > 0:
#     xaxis = np.linspace( 0, numTraces, numTraces )
#     jArr = np.array( jTrace )
#     alphaArr = np.array( alphaTrace )
#     axs[2].plot( xaxis, jArr, "red", label="j" )
#     axs[3].plot( xaxis, alphaArr, "green", label="alpha" )

plt.legend()
plt.show()




