import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from gradientdescent import *

LAMBDA = 0.00001

def ReadExperimentalData():

    m = 1000
    x = np.zeros((m, 4))
    y = np.zeros((m, 1))

    numInside = 0

    for i in range( m ):
        ang = np.random.random_sample() * 2.0 * np.pi
        radius = math.sqrt( np.random.random_sample() ) * math.sqrt( 2.0 )
        px = radius * math.cos( ang )
        py = radius * math.sin( ang )

        x[i][0] = px
        x[i][1] = py
        x[i][2] = px * px
        x[i][3] = py * py

        if ( math.sqrt( (px * px) + (py * py) ) <= 1.0 ):
            y[i][0] = 1.0
            numInside = numInside + 1

    # print( numInside )
    return (x,y)

def DrawExperimentalData( x, y ):

    # plt.figure()
    fig, axs = plt.subplots( 2, 1)

    xaxis = x[:,1]
    xaxis = xaxis[ y[:,0] > 0 ]
    yaxis = x[:,2]
    yaxis = yaxis[ y[:,0] > 0 ]

    axs[0].plot( xaxis, yaxis, "og", label="y" )
    axs[0].axis( 'equal')

    # The points outside
    xaxis = x[:,1]
    xaxis = xaxis[ y[:,0] < 1 ]
    yaxis = x[:,2]
    yaxis = yaxis[ y[:,0] < 1 ]

    axs[1].plot( xaxis, yaxis, "or", label="y" )
    axs[1].axis( 'equal')

    plt.legend()
    plt.show()

def ComputeH( x, theta ):
    # returns ( m x [1+n] ) x ( [n+1] x 1 )  = (mx1) matrix
    z = np.matmul( x, theta )
    h = 1 / ( 1 + np.exp( -z ) )
    return h

# def ComputeJ( y, h, l, theta ):
#     m = y.shape[0]
#     j = -(1/m) * ( np.matmul( y.transpose(), np.log(h) ) + np.matmul( (1-y).transpose(), np.log( 1-h ) ) )
#     j = j + ( (1/(2*m)) * ( np.matmul( l.transpose(), theta*theta)) )
#     return j

def ComputeJ( x, theta, y ):
    # y, h: (mx1)
    h = ComputeH( x, theta )
    j = -(1/m) * ( np.matmul( y.transpose(), np.log(h) ) + np.matmul( (1-y).transpose(), np.log( 1-h ) ) )
    return j

# Read input as tuple of np arrays
# input = ReadHousingData()
input = ReadExperimentalData()

# Set up x and y
x = input[0]
y = input[1]
CheckDimensions( x, y )

# Scale and normalise input
# ret = ScaleAndNormalise( x )
# x = ret[0]
# meanAndSum = ( ret[1], ret[2] )
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

# print (x, y)
# print (xTest, yTest)
DrawExperimentalData( x, y )
DrawExperimentalData( xTest, yTest )

# Set up initial theta
theta = np.zeros( (n+1, 1) )

# print( "x=\n", x)
# print( "y=\n", y)

theta = GradientDescent( x, y, theta, ComputeJ, ComputeH )
#
# # print( numIter, "iterations" )
# # print ( "error = ", j )
#
# j = ComputeJ( xTest, theta, yTest, xTest.shape[0] )
# print( "result error = ", j)
#
# fig, axs = plt.subplots( 4, 1)
#
# xaxis = x[:, 1]
# result = np.matmul( x, theta )
# axs[0].plot( xaxis, y, "b.", label="y" )
# axs[0].plot( xaxis, result, "orange",  label = "result" )
#
# sortIdx = xTest[:,1].argsort()
# xTest = xTest[ sortIdx ]
# yTest = yTest[ sortIdx ]
# xaxis = xTest[:, 1]
#
# result = np.matmul( xTest, theta )
# axs[1].plot( xaxis, yTest, "g.", label="test")
# axs[1].plot( xaxis, result, "orange",  label = "result" )
#
#
# # if numTraces > 0:
# #     xaxis = np.linspace( 0, numTraces, numTraces )
# #     jArr = np.array( jTrace )
# #     alphaArr = np.array( alphaTrace )
# #     axs[2].plot( xaxis, jArr, "red", label="j" )
# #     axs[3].plot( xaxis, alphaArr, "green", label="alpha" )
#
# plt.legend()
# plt.show()




