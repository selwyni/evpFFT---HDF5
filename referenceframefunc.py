import numpy as np

def euler2quaternion(phi1, Phi, phi2, P = 1):
    # Input - Euler Angles in Radians, Permutation operator (+- 1)
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
    # Output - Tuple of Euler Angles in Radians in Bunge Convention
    q03 = np.square(q0) + np.square(q3)
    q12 = np.square(q1) + np.square(q2)
    CHI = np.sqrt(q03*q12)

    if (CHI == 0 and q12 == 0):
        THETA = (np.arctan2(-2*P*q0*q3, q0^2 - q3^2), 0, 0)
    elif (CHI == 0 and q03 == 0):
        THETA = (np.arctan2(2*q1*q2, q1^2 - q2^2), np.pi, 0)
    elif (CHI != 0):
        THETA01 = (q1*q3 - P*q0*q2) / CHI
        THETA02 = (-P*q0*q1 - q2*q3) / CHI
        THETA11 = 2*CHI
        THETA12 = q03 - q12
        THETA21 = (P*q0*q2 + q1*q3) / CHI
        THETA22 = (q2*q3 - P*q0*q1) / CHI

        THETA = []
        for angle in [(THETA01, THETA02), (THETA11, THETA12), (THETA21, THETA22)]:
            val = np.arctan2(angle[0], angle[1])
            if (val < 0):
                val += 2*np.pi
            THETA.append(val)

        THETA = tuple(THETA)
    return THETA


def euler2orimatrix(phi1, Phi, phi2, P = 1):
    # Input - Euler Angle in Radians, Permutation operator (+- 1)
    # Output - Numpy orientation matrix
    C1 = np.cos(phi1)
    C2 = np.cos(phi2)
    S1 = np.sin(phi1)
    S2 = np.sin(phi2)
    C = np.cos(Phi)
    S = np.sin(Phi)

    E1 = C1*C2 - S1*C*S2
    E2 = S1*C2 - C1*C*S2
    E3 = S*S2
    E4 = -C1*S2 - S1*C*C2
    E5 = -S1*S2 + C1*C*C2
    E6 = S*C2
    E7 = S1*S
    E8 = -C1*S
    E9 = C

    return np.matrix([[E1, E2, E3], [E4, E5, E6] , [E7, E8, E9]])



def euler2axisangle(phi1, Phi, phi2, P = 1):
    # Input - Euler Angles in Radians, Permutation operator (+- 1)
    # Output - Tuple containing (axis1, axis2, axis3, angle)
    T = np.tan(Phi / 2)
    SIGMA = (1/2)*(phi1 + phi2)
    DELTA = (1/2)*(phi1 - phi2)
    TAU = np.sqrt(np.square(T) + np.square(np.sin(SIGMA)))
    OMEGA = 2*np.arctan(TAU / np.cos(SIGMA))
    if (OMEGA > np.pi):
        OMEGA = 2*np.pi - OMEGA

    axis1 = (P/TAU)*T*np.cos(DELTA)
    axis2 = (P/TAU)*T*np.sin(DELTA)
    axis3 = (P/TAU)*T*np.sin(SIGMA)

    return (axis1, axis2, axis3, OMEGA)


def orimatrix2euler(mat, P = 1):
    # Input - Numpy 3x3 Orientation Matrix
    # Output - Tuple of Euler Angles in Radians
    ZETA = (1 / np.sqrt(1 - np.square(mat[2,2])))
    if (mat[2,2] == 1):
        THETA1 = np.arctan2(mat[0,1], mat[0,0])
        THETA2 = (np.pi/2)*(1 - mat[2,2])
        THETA3 = 0
    else:
        THETA1 = np.arctan2(mat[2,0]*ZETA, -(mat[2,1]*ZETA))
        THETA2 = np.arccos(mat[2,2])
        THETA3 = np.arctan2(mat[0,2]*ZETA, mat[1,2]*ZETA)

    return (THETA1, THETA2, THETA3)

def orimatrix2quaternion(mat, P = 1):
    # Input - Numpy 3x3 Orientation Matrix
    # Output - Tuple containing (q0, q1, q2, q3)
    q0 = (1/2)*np.sqrt(1 + mat[0,0] + mat[1,1] + mat[2,2])
    q1 = (P/2)*np.sqrt(1 + mat[0,0] - mat[1,1] - mat[2,2])
    q2 = (P/2)*np.sqrt(1 - mat[0,0] + mat[1,1] - mat[2,2])
    q3 = (P/2)*np.sqrt(1 - mat[0,0] - mat[1,1] + mat[2,2])

    if (mat[2,1] < mat[1,2]):
        q1 = -q1

    if (mat[0,2] < mat[2,0]):
        q2 = -q2

    if (mat[1,0] > mat[0,1]):
        q3 = -q3

    MAGNITUDE = np.sqrt(np.square(q0) + np.square(q1) + np.square(q2) + np.square(q3))

    return (q0/MAGNITUDE, q1/MAGNITUDE, q2/MAGNITUDE, q3/MAGNITUDE)

#TODO: Quaternion misorientation
#TODO: GAM
#TODO: Find Symmetry operator quatonion form

def quaternion2axisangle(q0, q1, q2, q3, P = 1):
    # Input - Four quaternion values
    # Output - Tuple containing a list of three axis values and an angle in radians
    OMEGA = 2*np.arccos(q0)

    if OMEGA == 0:
        return ([q1, q2, q3], np.pi)
    else:
        s = np.sign(q0) / np.sqrt(np.square(q1) + np.square(q2) + np.square(q3))
        return ([s*q1, s*q2, s*q3], OMEGA)
