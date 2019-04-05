import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def ReadHousingData():
    # Load csv file
    df = pd.read_csv("house_data_small.csv")

    x = df[["sq_feet", "num_bedrooms", "num_bathrooms"]].values
    y = df[["sale_price"]].values

    return ( x, y )

def ReadExperimentalData():
    x = np.linspace(0, 100, 100)
    x.shape = (100, 1)

    y = x * x * x

    return ( x, y )


def ScaleAndNormalise( v ):

    mean = np.mean(v, axis=0)
    sum = np.sum(v, axis=0)

    v = np.subtract(v, mean)
    v = np.divide(v, sum)

    return (v, mean, sum )


# Read input as tuple of np arrays
# input = ReadHousingData()
input = ReadExperimentalData()

# Set up x and y
x = input[0]
y = input[1]

m = x.shape[0]
n = x.shape[1]

# Scale and normalise input
# ret = ScaleAndNormalise( x )
# x = ret[0]
# meanAndSum = ( ret[1], ret[2] )

# Insert the leading column of 1's
x = np.insert( x, 0, 1, axis=1 )

print( "x=", x)
print( "y=", y)

# Gradient descent
theta = np.zeros( (n+1, 1) )

for i in range( 0, 3000 ):
    w = np.matmul( x.transpose(), np.matmul( x, theta) - y )
    theta = theta - (0.0001/m) * w

print("theta = ", theta)

result = np.matmul( x, theta )
print ( "result =", result )

plt.figure()
xaxis = x[:, 1]
plt.plot( xaxis, y, label="y" )
plt.plot( xaxis, result, label = "result" )
plt.legend()
plt.show()




