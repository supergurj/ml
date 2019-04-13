import numpy as np
s = np.linspace(0, 4, 4)
print( s )
s.shape = ( 4, 1 )
print ( s)
s.shape = ( 1, 4 )
print( s )
s.shape = ( 4, )
print( s )

t = np.dot ( s , s)
print ( t )