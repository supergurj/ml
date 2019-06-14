import numpy as np

ALPHA_START = 10.0
ALPHA_SCALE = 0.5
ALPHA_MIN = 1.0e-4
TERM_MAX_ITERATIONS = 10000000
TERM_MIN_ERROR_RELATIVE_DELTA = 1.0e-4

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

        print ( row )

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

def GradientDescent( x, y, theta, regLambda, ComputeJ ):

    CheckDimensions( x, y )

    assert len(theta.shape) == 2
    assert x.shape[1] == theta.shape[0]
    assert theta.shape[1] == 1

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
    traceInterval = 1000 # sample data each time after this many iterations

    numIter = 0
    while 1:

        ll = 1 - alpha * l
        h = ComputeH( x, theta )    # Linear Regression: np.matmul( x, theta)
        w = np.matmul( x.transpose(), h - y )
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