import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import math

x=np.linspace(0,10,100)
x.shape=(100,1)

y=x*x*x

# Perturb
z=np.linspace(0,720,100)*(math.pi/180.0)
z=np.sin(z)*100.0
z.shape = ( 100, 1 )

y = y + z

plt.figure()
plt.plot( x, y )
plt.plot( x, z )
plt.show()