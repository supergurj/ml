import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

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

    y = x * x * x

    # Perturb
    z = np.linspace( 0, 720, m ) * ( math.pi / 180.0 )
    z = np.sin( z ) * 0.75
    z.shape = ( m, 1 )

    y = y + z

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

# # Split data into training and testing sets
# splitData = SplitDatasetIntoTrainingAndTest( x, y, 20 )
#
# x = splitData[0]
# y = splitData[1]
# CheckDimensions( x, y )
# m = x.shape[0]
# n = x.shape[1]
#
# xTest = splitData[2]
# yTest = splitData[3]
# CheckDimensions( xTest, yTest )
# assert xTest.shape[1] == n

# Insert the leading column of 1's
x = np.insert( x, 0, 1, axis=1 )
# xTest = np.insert( xTest, 0, 1, axis=1 )

# Set up initial theta
theta = np.zeros( (n+1, 1) )

# print( "x=\n", x)
# print( "y=\n", y)

# Gradient descent

ALPHA_START = 10.0
ALPHA_SCALE = 0.5
ALPHA_MIN = 1.0e-4
TERM_MAX_ITERATIONS = 10000000
TERM_MIN_ERROR_RELATIVE_DELTA = 1.0e-6

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

fig, axs = plt.subplots( 2, 1)


xaxis = x[:, 1]

axs[0].plot( xaxis, y, "b.", label="y" )
axs[0].plot( xaxis, result, "orange",  label = "result" )

# xaxis = xTest[:, 1]
# result = np.matmul( xTest, theta )
# axs[1].plot( xaxis, yTest, "g.", label="test")
# axs[1].plot( xaxis, result, "orange",  label = "result" )

plt.legend()
plt.show()




