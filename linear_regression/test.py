import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

m = 10
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

print( numInside )
print ( x )
print ( y )

xaxis = x[:, 0]
print ( xaxis )

print( xaxis[ y[:,0] > 0 ] )
