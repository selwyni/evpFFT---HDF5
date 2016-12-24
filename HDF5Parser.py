import sys
import os

from referenceframefunc import *
print(sys.version)
sys.path.append('/usr/local/lib/python3.5/dist-packages')
import numpy as np
from scipy import stats
import h5py
from itertools import product

####################
# SET PRIOR TO USE
####################
CWD = ''
os.chdir(CWD)




def readHDF5(filename, permissions='r'):
    sample = h5py.File(filename, permissions)
    container = sample['3Ddatacontainer']
    return container


################################################
# Dataset Retrieval Functions
################################################

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
    # Output - Tuple with average phi1, Phi, ph2 in Bunge Convention
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

    THETA = quaternion2euler(meanq0, meanq1, meanq2, meanq3)
    return THETA

################################################
# Writing Functions
################################################

def writeMeanSVMandEVM(filename):
    sampledata = readHDF5(filename, 'r+')
    datapointdirs = retrieveDatapoints(sampledata)
    dimensions = retrieveDimension(sampledata)
    grainIDs = retrieveGrainIDs(sampledata)
    numOfGrains = np.nanmax(grainIDs)
    phases = retrievePhase(sampledata)

    SVMs = []
    EVMs = []

    avgmeanSVM = []
    avgmeanEVM = []

    allAvgSVM = []
    allAvgEVM = []

    BCCSVM = []
    BCCEVM = []
    HCPSVM = []
    HCPEVM = []

    sigmaSVMs = []
    sigmaEVMs = []
    maxSVMs = []
    maxEVMs = []
    minSVMs = []
    minEVMs = []
    medianSVMs = []
    medianEVMs = []

    grainvolumes = []
    slipsys = []
    bungephi1 = []
    bungePhi = []
    bungephi2 = []
    for step in range(len(datapointdirs)):
        print("Going through Step", step)
        SVM = retrieveSVM(datapointdirs[step], dimensions, 'SVM')
        EVM = retrieveEVM(datapointdirs[step], dimensions, 'EVM')
        slip = retrieveSlipInformation(datapointdirs[step], dimensions)
        Phi = retrieveEulerAngles(datapointdirs[step], dimensions, 'Phi')
        phi1 = retrieveEulerAngles(datapointdirs[step], dimensions, 'phi1')
        phi2 = retrieveEulerAngles(datapointdirs[step], dimensions,'phi2')

        meanSVM = []
        meanEVM = []
        sigmaSVM = []
        sigmaEVM = []
        maxSVM = []
        maxEVM = []
        minSVM = []
        minEVM = []
        medianSVM = []
        medianEVM = []
        grainsize = []
        stepslipsys = []
        grainphi1 = []
        grainPhi = []
        grainphi2 = []
        for grainID in np.arange(1, numOfGrains + 1):
            # For the properties of individual grains.
            # Output is a list of 1 value per grain
            condition = grainIDs == int(grainID)
            grainSVM = np.extract(condition, SVM)
            grainEVM = np.extract(condition, EVM)
            grainslip = np.extract(condition, slip)
            grainPhiSet = np.extract(condition, Phi)
            grainPhi1Set = np.extract(condition, phi1)
            grainPhi2Set = np.extract(condition, phi2)
            (phi1val, Phival, phi2val) = grainAverageEulerAngle(grainPhi1Set, grainPhiSet, grainPhi2Set)

            meanSVM.append(np.mean(grainSVM))
            meanEVM.append(np.mean(grainEVM))

            sigmaSVM.append(np.std(grainSVM))
            sigmaEVM.append(np.std(grainEVM))

            maxSVM.append(np.max(grainSVM))
            maxEVM.append(np.max(grainEVM))

            minSVM.append(np.min(grainSVM))
            minEVM.append(np.min(grainEVM))

            medianSVM.append(np.median(grainSVM))
            medianEVM.append(np.median(grainEVM))

            grainsize.append(np.sum(condition))
            stepslipsys.append(np.mean(grainslip))

            grainphi1.append(phi1val)
            grainPhi.append(Phival)
            grainphi2.append(phi2val)

        for phase in [1,2]:
            # Pick out phase properties
            condition = phases == phase
            if (phase == 1):
                BCCSVMvals = np.extract(condition, SVM)
                BCCEVMvals = np.extract(condition, EVM)
                BCCSVM.append(np.mean(BCCSVMvals))
                BCCEVM.append(np.mean(BCCEVMvals))
            else:
                HCPSVMvals = np.extract(condition,SVM)
                HCPEVMvals = np.extract(condition,EVM)
                HCPSVM.append(np.mean(HCPSVMvals))
                HCPEVM.append(np.mean(HCPEVMvals))

        # Aggregating List of Grain by Grain properties
        SVMs.append(meanSVM)
        EVMs.append(meanEVM)

        sigmaSVMs.append(sigmaSVM)
        sigmaEVMs.append(sigmaEVM)

        maxSVMs.append(maxSVM)
        maxEVMs.append(maxEVM)

        minSVMs.append(minSVM)
        minEVMs.append(minEVM)

        medianSVMs.append(medianSVM)
        medianEVMs.append(medianEVM)

        grainvolumes.append(grainsize)
        slipsys.append(stepslipsys)

        bungephi1.append(grainphi1)
        bungePhi.append(grainPhi)
        bungephi2.append(grainphi2)

        # Grain weighted properties
        avgmeanSVM.append(np.mean(meanSVM))
        avgmeanEVM.append(np.mean(meanEVM))

        allAvgSVM.append(np.mean(SVM))
        allAvgEVM.append(np.mean(EVM))

    allPoints = np.transpose(np.array([allAvgSVM, allAvgEVM]))
    avgmat = np.transpose(np.array([avgmeanSVM, avgmeanEVM]))
    SVMmat = np.transpose(np.array(SVMs))
    EVMmat = np.transpose(np.array(EVMs))
    sigmaSVMmat = np.transpose(np.array(sigmaSVMs))
    sigmaEVMmat = np.transpose(np.array(sigmaEVMs))
    maxSVMmat = np.transpose(np.array(maxSVMs))
    maxEVMmat = np.transpose(np.array(maxEVMs))
    minSVMmat = np.transpose(np.array(minSVMs))
    minEVMmat = np.transpose(np.array(minEVMs))
    medianSVMmat = np.transpose(np.array(medianSVMs))
    medianEVMmat = np.transpose(np.array(medianEVMs))
    BCCphasemat = np.transpose(np.array([BCCSVM, BCCEVM]))
    HCPphasemat = np.transpose(np.array([HCPSVM, HCPEVM]))
    grainsizemat = np.transpose(np.array(grainvolumes))
    slipmat = np.transpose(np.array(slipsys))
    print(np.shape(slipmat))
    phi1mat = np.transpose(np.array(bungephi1))
    Phimat = np.transpose(np.array(bungePhi))
    phi2mat = np.transpose(np.array(bungephi2))

    if ('MeanSVM' not in sampledata):
        sampledata.create_dataset("MeanSVM", data=SVMmat)
    if ('MeanEVM' not in sampledata):
        sampledata.create_dataset("MeanEVM", data=EVMmat)
    if ('sigmaSVM' not in sampledata):
        sampledata.create_dataset("sigmaSVM", data = sigmaSVMmat)
    if ('sigmaEVM' not in sampledata):
        sampledata.create_dataset("sigmaEVM", data = sigmaEVMmat)
    if ('maxSVM' not in sampledata):
        sampledata.create_dataset("maxSVM", data = maxSVMmat)
    if ('maxEVM' not in sampledata):
        sampledata.create_dataset("maxEVM", data = maxEVMmat)
    if ('minSVM' not in sampledata):
        sampledata.create_dataset("minSVM", data = minSVMmat)
    if ('minEVM' not in sampledata):
        sampledata.create_dataset("minEVM", data = minEVMmat)
    if ('medianSVM' not in sampledata):
        sampledata.create_dataset("medianSVM", data = medianSVMmat)
    if ('medianEVM' not in sampledata):
        sampledata.create_dataset("medianEVM", data = medianEVMmat)
    if ('StepAverages' not in sampledata):
        sampledata.create_dataset("StepAverages", data=avgmat)
    if ('AllPoints' not in sampledata):
        sampledata.create_dataset("AllPoints", data=allPoints)
    if ('MeanBCCAvgs' not in sampledata):
        sampledata.create_dataset("MeanBCCAvgs", data=BCCphasemat)
    if ('MeanHCPAvgs' not in sampledata):
        sampledata.create_dataset("MeanHCPAvgs", data = HCPphasemat)
    if ('sigmaSVM' not in sampledata):
        sampledata.create_dataset("sigmaSVM", data = sigmaSVMmat)
    if ('grainVolume' not in sampledata):
        sampledata.create_dataset("grainVolume", data = grainsizemat)
    if ('avgSlipSys' not in sampledata):
        sampledata.create_dataset('avgSlipSys', data = slipmat)
    if ('grainAvgphi1' not in sampledata):
        sampledata.create_dataset('grainAvgphi1', data = phi1mat)
    if ('grainAvgPhi' not in sampledata):
        sampledata.create_dataset('grainAvgPhi', data = Phimat)
    if ('grainAvgphi2' not in sampledata):
        sampledata.create_dataset('grainAvgphi2', data = phi2mat)

def writeMeansToCSV(filename):
    sampledata = readHDF5(filename, 'r+')
    stepcount = 0
    grainIDs = retrieveGrainIDs(sampledata)
    numOfGrains = int(np.nanmax(grainIDs))

    if (not ('MeanSVM' in sampledata and
        'MeanEVM' in sampledata and
        'StepAverages' in sampledata and
        'AllPoints' in sampledata and
        'MeanBCCAvgs' in sampledata and
        'MeanHCPAvgs' in sampledata and
        'sigmaSVM' in sampledata and
        'sigmaEVM' in sampledata and
        'maxSVM' in sampledata and
        'maxEVM' in sampledata and
        'minSVM' in sampledata and
        'minEVM' in sampledata and
        'medianSVM' in sampledata and
        'medianEVM' in sampledata and
        'grainVolume' in sampledata and
        'avgSlipSys' in sampledata and
        'grainAvgphi1' in sampledata and
        'grainAvgPhi' in sampledata and
        'grainAvgphi2' in sampledata)):
        writeMeanSVMandEVM(filename)

    for step in sampledata:
        if ('Step-' in step):
            stepcount += 1


    SVMarr = np.zeros((numOfGrains, stepcount))
    EVMarr = np.zeros((numOfGrains, stepcount))
    sigmaSVMarr = np.zeros((numOfGrains, stepcount))
    sigmaEVMarr = np.zeros((numOfGrains, stepcount))
    maxSVMarr = np.zeros((numOfGrains, stepcount))
    maxEVMarr = np.zeros((numOfGrains, stepcount))
    minSVMarr = np.zeros((numOfGrains, stepcount))
    minEVMarr = np.zeros((numOfGrains, stepcount))
    medianSVMarr = np.zeros((numOfGrains, stepcount))
    medianEVMarr = np.zeros((numOfGrains, stepcount))
    grainVolumearr = np.zeros((numOfGrains, stepcount))
    sliparr = np.zeros((numOfGrains, stepcount))
    phi1arr = np.zeros((numOfGrains, stepcount))
    Phiarr = np.zeros((numOfGrains, stepcount))
    phi2arr = np.zeros((numOfGrains, stepcount))
    avgarr = np.zeros((stepcount, 2))
    allavgarr = np.zeros((stepcount, 2))
    BCCarr = np.zeros((stepcount, 2))
    HCParr = np.zeros((stepcount,2))

    sampledata['MeanSVM'].read_direct(SVMarr)
    sampledata['MeanEVM'].read_direct(EVMarr)
    sampledata['sigmaSVM'].read_direct(sigmaSVMarr)
    sampledata['sigmaEVM'].read_direct(sigmaEVMarr)
    sampledata['maxSVM'].read_direct(maxSVMarr)
    sampledata['maxEVM'].read_direct(maxEVMarr)
    sampledata['minSVM'].read_direct(minSVMarr)
    sampledata['minEVM'].read_direct(minEVMarr)
    sampledata['medianSVM'].read_direct(medianSVMarr)
    sampledata['medianEVM'].read_direct(medianEVMarr)
    sampledata['StepAverages'].read_direct(avgarr)
    sampledata['AllPoints'].read_direct(allavgarr)
    sampledata['MeanBCCAvgs'].read_direct(BCCarr)
    sampledata['MeanHCPAvgs'].read_direct(HCParr)
    sampledata['grainVolume'].read_direct(grainVolumearr)
    sampledata['avgSlipSys'].read_direct(sliparr)
    sampledata['grainAvgphi1'].read_direct(phi1arr)
    sampledata['grainAvgPhi'].read_direct(Phiarr)
    sampledata['grainAvgphi2'].read_direct(phi2arr)

    topname = filename.split('.')[0]

    np.savetxt(topname + 'GrainMeanSVM.csv', SVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMeanEVM.csv', EVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainSigmaSVM.csv', sigmaSVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainSigmaEVM.csv', sigmaEVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMaxSVM.csv', maxSVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMaxEVM.csv', maxEVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMinSVM.csv', minSVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMinEVM.csv', minEVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMedianSVM.csv', medianSVMarr, delimiter = ',')
    np.savetxt(topname + 'GrainMedianEVM.csv', medianEVMarr, delimiter = ',')
    np.savetxt(topname + 'TimeStepGrainAvg.csv', avgarr, delimiter = ',')
    np.savetxt(topname + 'TimeStepVolumeAvg.csv', allavgarr, delimiter = ',')
    np.savetxt(topname + 'TimeBCCAvg.csv', BCCarr, delimiter = ',')
    np.savetxt(topname + 'TimeHCPAvg.csv', HCParr, delimiter = ',')
    np.savetxt(topname + 'GrainVolume.csv', grainVolumearr, delimiter = ',')
    np.savetxt(topname + 'SlipSystems.csv', sliparr, delimiter = ',')
    np.savetxt(topname + 'Phi1Angle.csv', phi1arr, delimiter = ',')
    np.savetxt(topname + 'PhiAngle.csv', Phiarr, delimiter = ',')
    np.savetxt(topname + 'Phi2Angle.csv', phi2arr, delimiter = ',')


writeMeansToCSV('f20_eqdata.h5')
