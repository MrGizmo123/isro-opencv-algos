
# These are the formulas calculated in the pdf

def theta(x, y, gamma1 = 31.0, gamma2 = 37.0, w = 640.0, h = 480.0):

    x = x - w/2.0; # make it so that the center of the image has coords 0,0
    y = y - h/2.0;
    
    gamma1 = np.deg2rad(gamma1);# convert from degree to radian
    gamma2 = np.deg2rad(gamma2);
    
    tan1 = np.tan(gamma1/2.0)
    tan2 = np.tan(gamma2/2.0)

    x_scaled = x/w
    y_scaled = y/h

    summation = (x_scaled ** 2) * (tan2 ** 2) + (y_scaled ** 2) * (tan1 ** 2)

    second_last = 2 * np.sqrt(summation)

    result = np.atan(second_last)

    return np.rad2deg(result) # radian to degrees

def phi(x, y, gamma1 = 31.0, gamma2 = 37.0, w = 640.0, h = 480.0):

    x = x - w/2.0;              # make it so that the center of the image has coords 0,0
    y = y - h/2.0;
    
    tan1 = np.tan(gamma1/2.0)   # convert from degree to radian
    tan2 = np.tan(gamma2/2.0)
    
    numerator = w * y * tan1
    denominator = h * x * tan2

    result = np.atan2(numerator, denominator)

    return np.rad2deg(result)   # radian to degrees
