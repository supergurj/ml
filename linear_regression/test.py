import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

def CheckDimensions( x, y ):

    assert len( x.shape ) == 2
    assert len( y.shape ) == 2
    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == 1
    return

def ReadExperimentalData():

    maxX = 10
    numSteps = 50

    x = np.linspace(0, maxX, numSteps)
    x.shape = (numSteps, 1)

    y = x

    # # Perturb
    # z = np.linspace( 0, 720, numSteps ) * ( math.pi / 180.0 )
    # z = np.sin( z ) * 0.25
    # z.shape = ( maxX, 1 )

    # y = y + z
    #
    # # Add in higher orders of x
    # xOrig = x
    xx = x
    for i in range ( 0, 5 ):
        # xx = xx * xOrig
        x = np.append( x, xx, axis=1 )

    return ( x, y )

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

        xTest[i:i+1:,] = xTraining[row:row+1:,]
        yTest[i:i+1:,] = yTraining[row:row+1:,]

        xTraining = np.delete( xTraining, row, 0 )
        yTraining = np.delete( yTraining, row, 0 )

        m = xTraining.shape[0]

    return ( xTraining, yTraining, xTest, yTest )


inputData = ReadExperimentalData()

# Set up x and y
x = inputData[0]
y = inputData[1]

CheckDimensions( x, y )

# Split data into training and testing sets
splitData = SplitDatasetIntoTrainingAndTest( x, y, 50 )

x = splitData[0]
y = splitData[1]
CheckDimensions( x, y )

xTest = splitData[2]
yTest = splitData[3]
CheckDimensions( xTest, yTest )

print( xTest )
print( xTest[ xTest[:,0].argsort()] )

plt.figure()

# print("theta = ", theta)
# print ( "result =", result )
# print( xaxis )

plt.plot( x[:, 0], y, "b.", label="y" )
plt.plot( xTest[:,0], yTest, "go", label="test")

plt.legend()
plt.show()