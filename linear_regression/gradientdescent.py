import numpy as np

ALPHA_START = 10.0
ALPHA_SCALE = 0.5
ALPHA_MIN = 1.0e-4
TERM_MAX_ITERATIONS = 10000000
TERM_MIN_ERROR_RELATIVE_DELTA = 1.0e-4

def GradientDescent( x, y, theta, regLambda, ComputeJ ):

    m = x.shape[0]
    n = x.shape[1] - 1

    # Set up lambda
    l = np.ones( (n+1, 1) )
    l[0][0] = 0
    l = (regLambda/m) * l

    j = ComputeJ( x, theta, y, m )
    alpha = ALPHA_START

    jTrace = []
    alphaTrace = []
    numTraces = 0
    traceInterval = 1000 # sample data each time after this many interations

    numIter = 0
    while 1:

        ll = 1 - alpha * l
        w = np.matmul( x.transpose(), np.matmul( x, theta) - y )
        thetaNew = (ll * theta) - ( (alpha/m) * w )
        jNew = ComputeJ( x, thetaNew, y, m )

        if ( jNew < j ):

            # Update theta
            theta = thetaNew

            # Check for termination
            if (  jNew < TERM_MIN_ERROR_RELATIVE_DELTA ) :
                print( "Terminating after relative error decreasing below threshold.")
                break

            if ( numIter > TERM_MAX_ITERATIONS ) :
                print( "Terminating after %d iterations." % numIter )
                break

            alpha = alpha * 1.1

        else:

            if ( alpha < ALPHA_MIN ):
                # Time to quit
                print( "Terminating after relative error started increasing with min alpha.")
                break
            else:
                # Try smaller step
                alpha = alpha * 0.5

        j = jNew
        numIter = numIter + 1

        if ( (numIter % traceInterval) == 0 ):
            jTrace.append( j )
            alphaTrace.append( alpha )
            numTraces = numTraces + 1

    return theta