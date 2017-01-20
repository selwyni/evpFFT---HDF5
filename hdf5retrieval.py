import numpy as np
from referenceframefunc import *

def retrieveSpacing(container):
    # Input - HDF5 Container
    # Output - Dict w/ spacing specs
    steplist = []
    LatticeParameters = []
    x = []
    y = []
    z = []
    for step in container:
        if ('Step' in step):
            steplist.append(container[step])
            spacing = container[step]['Geometry']['Spacing']
            x.append(spacing[0])
            y.append(spacing[1])
            z.append(spacing[2])

    spec = {'x': x, 'y':y, 'z':z}
    return spec

def retrieveDatapoints(container):
    # Input - HDF5 container
    # Output - List of Datapoint directories of each step in order.
    datapoints = []
    for step in container:
        if ('Step-' in step):
            datapoints.append(container[step]['Datapoint'])
    return datapoints

def retrieveDimension(container):
    # Input - HDF5 Container
    # Output - Dictionary of x, y, z axes
    for step in container:
        if ('Step' in step):
            dims = container[step]['Geometry']['Dimension']
            break
    spec = {'x':dims[0], 'y':dims[1], 'z':dims[2]}
    return spec


def retrieveGrainIDs(container):
    # Input - HDF5 container
    # Output - Numpy 3D array
    for step in container:
        if ('Step' in step):
            grainIDs = container[step]['Datapoint']['Grainid']
            break
    dims = retrieveDimension(container)
    arr = np.zeros((dims['x'], dims['y'], dims['z']))
    grainIDs.read_direct(arr)
    return arr

def retrievePhase(container):
    # Input - HDF5 container
    # Output - Numpy 3D array
    for step in container:
        if ('Step' in step):
            phase = container[step]['Datapoint']['Phase']
            break
    dims = retrieveDimension(container)
    arr = np.zeros((dims['x'], dims['y'], dims['z']))
    phase.read_direct(arr)
    return arr

def retrieveSVM(datapoint, dims, SVM='SVM'):
    # Input - Datapoint directory, dimensions, SVM dataset
    # SVM = ['Mean_Stress', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'SVM']
    # Output  Numpy 3D array of dataset
    arr = np.zeros((dims['x'], dims['y'], dims['z']))
    SVMs = datapoint['Sfields'][SVM]
    SVMs.read_direct(arr)
    return arr

def retrieveEVM(datapoint, dims, EVM='EVM'):
    # Input - Datapoint directory, dimensions, EVM dataset
    # EVM = ['D11', 'D12', 'D13', 'D22', 'D23', 'D33', 'EVM']
    # Output  Numpy 3D array of dataset
    arr = np.zeros((dims['x'], dims['y'], dims['z']))
    SVMs = datapoint['Dfields'][EVM]
    SVMs.read_direct(arr)
    return arr

def retrieveEulerAngles(datapoint, dims, angle='Phi'):
    # Input - Datapoint directory, dimensions, angle = ['Phi', 'Phi1', 'phi2']
    # Output - Numpy 3D array containing the particular EulerAngle
    if (angle.lower() == 'phi'):
        dataset = 'Phi'
    elif (angle.lower() == 'phi1'):
        dataset = 'Phi1'
    elif (angle.lower() == 'phi2'):
        dataset = 'phi2'
    arr = np.zeros((dims['x'], dims['y'], dims['z']))
    angle = datapoint['Eulerangle'][dataset]
    angle.read_direct(arr)
    return arr

def retrieveSlipInformation(datapoint, dims):
    # Input - Datapoint directory, dimensions
    # Output - Number 3D array containing slip information per voxel
    arr = np.zeros((dims['x'], dims['y'], dims['z']))
    slip = datapoint['Slip_information']['Slip_active_no']
    slip.read_direct(arr)
    return arr

def grainAverageEulerAngle(phi1, Phi, phi2, P = 1):
    # Input -  Bunge Convention phi1, Phi, phi2 arrays in DEGREES, Permutation operator (+- 1)
    # Output - Tuple with average phi1, Phi, ph2 in Radians, in Bunge Convention
    phi1 = np.radians(phi1)
    Phi = np.radians(Phi)
    phi2 = np.radians(phi2)

    q0vals = []
    q1vals = []
    q2vals = []
    q3vals = []

    # Convert to radians
    for obj in [phi1, Phi, phi2]:
        # phi1 in [0, 2*pi]
        # Phi in [0, pi]
        # phi2 in [0, 2*pi]
        condition = obj < 0
        obj[condition] += 2*np.pi

    assert(np.size(phi1) == np.size(phi2) == np.size(Phi))

    for idx in range(np.size(phi1)):
        (q0, q1, q2, q3) = euler2quaternion(phi1[idx], Phi[idx], phi2[idx])
        q0vals.append(q0)
        q1vals.append(q1)
        q2vals.append(q2)
        q3vals.append(q3)

    meanq0 = np.mean(q0vals)
    meanq1 = np.mean(q1vals)
    meanq2 = np.mean(q2vals)
    meanq3 = np.mean(q3vals)

    (q0, q1, q2, q3) = euler2quaternion(np.pi/2, 0, 0)

    THETA = quaternion2euler(meanq0, meanq1, meanq2, meanq3)
    return THETA

def grainOrientationSpread(avgMat, phi1List, PhiList, phi2List):
    # Input - Average Orientation Matrix, list of euler angles in radians to compare
    # Output - Single value of GOS
    misorAngles = []
    invAvgMat = inv(avgMat)
    for index in range(len(phi1List)):
        phi1 = phi1List[index]
        Phi = PhiList[index]
        phi2 = phi2List[index]
        matOfInterest = euler2orimatrix(phi1, Phi, phi2)
        delta = np.matmul(matOfInterest, invAvgMat)
        angle = np.arccos((np.trace(delta) - 1)/ 2)
        misorAngles.append(angle)
    return np.mean(misorAngles)
