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
X = df[["sq_feet", "num_bedrooms", "num_bathrooms"]].values
Y = df[["sale_price"]].values

# Scale and normalise input
# ret = ScaleAndNormalise( X )
# X = ret[0]
# X = np.insert( X, 0, 1, axis=1 )


print( "x=", X)
print( "y=", Y)

tx = X.transpose()
print ("x transpose = ", tx)

# for key, value in df.iteritems():
#
#     mean = value.mean()
#     range = value.max() - value.min()
#
#     print ( key, mean, range )
#
#     for i in range( dim[1] ):
#         print ( v[ cur, i ] + ' ')




