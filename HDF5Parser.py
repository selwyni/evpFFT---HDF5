import sys
import os

from referenceframefunc import *
print(sys.version)
sys.path.append('/usr/local/lib/python3.5/dist-packages')
import numpy as np
from scipy import stats
import h5py
from itertools import chain

####################
# SET PRIOR TO USE
####################
CWD = '/home/selwyni/Desktop/h5/TestData'
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

    GOS = []
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
        stepGOS = []
        for grainID in np.arange(1, numOfGrains + 1):
            # For the properties of individual grains.
            # Output is a list of 1 value per grain
            if (grainID % 100 = 0):
                print('\tGrain', grainID)

            condition = grainIDs == int(grainID)
            grainSVM = np.extract(condition, SVM)
            grainEVM = np.extract(condition, EVM)
            grainslip = np.extract(condition, slip)
            grainPhiSet = np.extract(condition, Phi)
            grainPhi1Set = np.extract(condition, phi1)
            grainPhi2Set = np.extract(condition, phi2)
            (phi1val, Phival, phi2val) = grainAverageEulerAngle(grainPhi1Set, grainPhiSet, grainPhi2Set)


            avgOriMatrix = euler2orimatrix(phi1val, Phival, phi2val)
            grainIDgos = grainOrientationSpread(avgOriMatrix, grainPhi1Set, grainPhiSet, grainPhi2Set)

            stepGOS.append(grainIDgos)

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

        GOS.append(stepGOS)
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


def writeDatasetToCSV(sampledata, h5datasetName, sizeOfArray, csvFilename):
    array = np.zeros(sizeOfArray)
    sampledata[h5datasetName].read_direct(array)
    np.savetxt(csvFilename, array, delimiter = ',')

def writeCCADataCSV(sampledata, numOfGrains, stepcount, datasets, steps, filename):
    for step in steps:
        writedata = np.arange(1, numOfGrains + 1)
        header = 'GrainIDs,'
        for dataset in datasets:
            header = header + dataset + ','
            dataArr = np.zeros((numOfGrains, stepcount))
            sampledata[dataset].read_direct(dataArr)
            writedata = np.vstack((writedata, dataArr[:,step]))

        writedata = np.transpose(writedata)
        np.savetxt(filename + 'Step' + str(step) + '.csv', writedata, delimiter = ',', header = header, comments='')




def writeMeansToCSV(filename):
    sampledata = readHDF5(filename, 'r+')
    stepcount = 0
    grainIDs = retrieveGrainIDs(sampledata)
    numOfGrains = int(np.nanmax(grainIDs))

    for step in sampledata:
        if ('Step-' in step):
            stepcount += 1

    datasetNames = [['MeanSVM', 'MeanEVM', 'sigmaSVM', 'sigmaEVM', 'maxSVM', 'maxEVM', 'minSVM', 'minEVM', 'medianSVM', 'medianEVM', 'grainVolume', 'avgSlipSys', 'grainAvgphi1', 'grainAvgPhi', 'grainAvgphi2'],
    ['StepAverages', 'AllPoints', 'MeanBCCAvgs', 'MeanHCPAvgs']]

    for name in [item for sublist in datasetNames for item in sublist]:
        if (name not in sampledata):
            writeMeanSVMandEVM(filename)

    topname = filename.split('.')[0]

    fileNames = [['GrainMeanSVM', 'GrainMeanEVM', 'GrainSigmaSVM', 'GrainSigmaEVM', 'GrainMaxSVM', 'GrainMaxEVM', 'GrainMinSVM', 'GrainMinEVM', 'GrainMedianSVM', 'GrainMedianEVM', 'GrainVolume', 'SlipSystems', 'Phi1Angle', 'PhiAngle', 'Phi2Angle'],
    ['TimeStepGrainAvg', 'TimeStepVolumeAvg', 'TimeBCCAvg', 'TimeHCPAvg']]

    for index in range(1):
        for dataset in range(len(datasetNames[index])):
            if (index == 0):
                arrshape = (numOfGrains, stepcount)
            elif (index == 1):
                arrshape = (stepcount, 2)

            writeDatasetToCSV(sampledata, datasetNames[index][dataset], arrshape, topname + fileNames[index][dataset] + '.csv')

    writeCCADataCSV(sampledata, numOfGrains, stepcount, ['grainVolume', 'grainAvgphi1', 'grainAvgPhi', 'grainAvgphi2', 'MeanSVM', 'MeanEVM'], [1, 5, 9], topname + 'CCA')


writeMeansToCSV('f20_eqdata.h5')
