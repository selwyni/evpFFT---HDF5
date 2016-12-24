import numpy

def euler2quaternion(phi1, Phi, phi2, P = 1):
    # Input - REQUIRES Euler Angles in Radians, Permutation operator (+- 1)
    # Output - Tuple containing quaternion form
    SIGMA = 0.5*(phi1 + phi2)
    DELTA = 0.5*(phi1 - phi2)
    C = np.cos(Phi/2)
    S = np.sin(Phi/2)

    q0 = C*np.cos(SIGMA)
    q1 = -P*S*np.cos(DELTA)
    q2 = -P*S*np.sin(DELTA)
    q3 = -P*C*np.sin(SIGMA)

    if (q0 < 0):
        return (-q0, -q1, -q2, -q3)
    else:
        return (q0, q1, q2, q3)

def quaternion2euler(q0, q1, q2, q3, P = 1):
    # Input - Quaternion, Permutation operator (+- 1)
    # Output - Tuple of Euler Angles in Bunge Convention
    q03 = q0^2 + q3^2
    q12 = q1^2 + q2^2
    CHI = np.sqrt(q03*q12)

    if (CHI = 0 and q12 = 0):
        THETA = (np.arctan2(-2*P*q0*q3, q0^2 - q3^2), 0, 0)
    elif (CHI = 0 and q03 = 0):
        THETA = (np.arctan2(2*q1*q2, q1^2 - q2^2), np.pi, 0)
    elif (CHI != 0):
        THETA01 = (q1*q3 - P*q0*q2) / CHI
        THETA02 = (-P*q0*q1 - q2*q3) / CHI
        THETA11 = 2*CHI
        THETA12 = q03 - q12
        THETA21 = (P*q0*q2 + q1*q3) / CHI
        THETA22 = (q2*q3 - P*q0*q1) / CHI

        THETA = (np.arctan2(THETA01, THETA02), np.arctan2(THETA11, THETA12), np.arctan2(THETA21, THETA22))

    return THETA
