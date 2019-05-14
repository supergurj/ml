import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

n = 5
m = 10

l = np.ones( (n+1, 1))
l[0][0] = 0
l = (1/m) * l

ll = 1 - 5 * l


print( l )
print( ll )

print ( l * ll )