import pandas as pd
import numpy as np

def ScaleAndNormalise( v ):

    mean = np.mean(v, axis=0)
    sum = np.sum(v, axis=0)

    v = np.subtract(v, mean)
    v = np.divide(v, sum)

    return (v, mean, sum )

# Load csv file
df = pd.read_csv("house_data_small.csv")

# Read data into dataframes
x = df[["sq_feet", "num_bedrooms", "num_bathrooms"]].values
m = np.size(x, 0)
n = np.size(x, 1)
y = df[["sale_price"]].values

# Scale and normalise input
ret = ScaleAndNormalise( x )
x = ret[0]
meanAndSum = ( ret[1], ret[2] )

# Insert the leading column of 1's
x = np.insert( x, 0, 1, axis=1 )

print( "x=", x)
print( "y=", y)

# Gradient descent
theta = np.zeros( (n+1, 1) )

for i in range( 0, 3000 ):
    w = np.matmul( x.transpose(), np.matmul( x, theta) - y )
    theta = theta - (0.1/m) * w

print("theta = ", theta)
print ( "result =", np.matmul( x, theta ))




