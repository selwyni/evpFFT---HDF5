import sys
import os
import csv
from referenceframefunc import *
from hdf5retrieval import *
print(sys.version)
sys.path.append('/usr/local/lib/python3.5/dist-packages')
import numpy as np
from scipy import stats
import h5py
from itertools import chain

####################
# SET PRIOR TO USE
####################
CWD = '/home/selwyni/Desktop/h5/Dec 20 Data'
os.chdir(CWD)


def readHDF5(filename, permissions='r'):
    sample = h5py.File(filename, permissions)
    container = sample['3Ddatacontainer']
    return container

#####################################
# Read DREAM3D CSVs
#####################################
def findCSVname(hdf5name):
    # Input - String of h5 filename
    # Output - String of csvfile path
    filename = hdf5name.split('data')[0]
    vol = int(filename[1:3]) // 10
    shape = filename.split('_')[1]
    shapetag = ''
    if (shape == '1051'):
        shapetag = '1051'
    elif (shape == 'eq'):
        shapetag = '111'
    elif (shape == 'disk'):
        shapetag = '10101'

    csvfilename = 'asp' + shapetag + '_vol0' + str(vol) + '.csv'
    return csvfilename

def retrieveDataFromCSV(csvfile, timesteps):
    # Input - String containing csv file
    # Output - Tuple of lists containing (q0, q1, q2, q3, surfaceareavolumeratio) for each
    with open(csvfile, 'r') as obj:
        reader = csv.reader(obj)
        q0 = []
        q1 = []
        q2 = []
        q3 = []
        shape = []
        for row in reader:
            if (row[0] == 'Feature_ID'):
                q0index = row.index('AvgQuats_0')
                q1index = row.index('AvgQuats_1')
                q2index = row.index('AvgQuats_2')
                q3index = row.index('AvgQuats_3')
                shapeindex = row.index('SurfaceAreaVolumeRatio')
                break
        for row in reader:
            q0.append(row[q0index])
            q1.append(row[q1index])
            q2.append(row[q2index])
            q3.append(row[q3index])
            shape.append(row[shapeindex])
        q0 = np.transpose(np.matrix(np.tile(np.array(q0, dtype = np.float32), (timesteps, 1))))
        q1 = np.transpose(np.matrix(np.tile(np.array(q1, dtype = np.float32), (timesteps, 1))))
        q2 = np.transpose(np.matrix(np.tile(np.array(q2, dtype = np.float32), (timesteps, 1))))
        q3 = np.transpose(np.matrix(np.tile(np.array(q3, dtype = np.float32), (timesteps, 1))))
        shape = np.transpose(np.matrix(np.tile(np.array(shape, dtype= np.float32), (timesteps, 1))))
        return (q0, q1, q2, q3, shape)

################################################
# Writing Functions
################################################

def writeDatasetToHDF5(filename):
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

    timesteps = len(datapointdirs)
    for step in range(timesteps):
        print("Going through Step", step)
        SVM = retrieveSVM(datapointdirs[step], dimensions, 'SVM')
        EVM = retrieveEVM(datapointdirs[step], dimensions, 'EVM')
        slip = retrieveSlipInformation(datapointdirs[step], dimensions)
        Phi = retrieveEulerAngles(datapointdirs[step], dimensions, 'Phi')
        phi1 = retrieveEulerAngles(datapointdirs[step], dimensions, 'phi1')
        phi2 = retrieveEulerAngles(datapointdirs[step], dimensions,'phi2')

        # TODO REFACTOR THIS
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
            if (grainID % 100 == 0):
                print('\tGrain', grainID)

            condition = grainIDs == int(grainID)
            grainSVM = np.extract(condition, SVM)
            grainEVM = np.extract(condition, EVM)
            grainslip = np.extract(condition, slip)
            grainPhiSet = np.extract(condition, Phi)
            grainPhi1Set = np.extract(condition, phi1)
            grainPhi2Set = np.extract(condition, phi2)
            (meanq0, meanq1, meanq2, meanq3) = grainAverageQuaternion(grainPhi1Set, grainPhiSet, grainPhi2Set)


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
    phi1mat = np.transpose(np.array(bungephi1))
    Phimat = np.transpose(np.array(bungePhi))
    phi2mat = np.transpose(np.array(bungephi2))
    (q0mat, q1mat, q2mat, q3mat, shapemat) = retrieveDataFromCSV(CWD + '/Undeformed/CSV/' + findCSVname(filename), timesteps)
    # TODO Find orientation, get difference in quaternion space
    # TODO REFACTOR THIS MESS
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
    if ('grainAvgQuat0' not in sampledata):
        sampledata.create_dataset('grainAvgQuat0', data = q0mat)
    if ('grainAvgQuat1' not in sampledata):
        sampledata.create_dataset('grainAvgQuat1', data = q1mat)
    if ('grainAvgQuat2' not in sampledata):
        sampledata.create_dataset('grainAvgQuat2', data = q2mat)
    if ('grainAvgQuat3' not in sampledata):
        sampledata.create_dataset('grainAvgQuat3', data = q3mat)
    if ('surfaceAreaVolumeRatio' not in sampledata):
        sampledata.create_dataset('surfaceAreaVolumeRatio', data = shapemat)

def writeDatasetToCSV(sampledata, h5datasetName, sizeOfArray, csvFilename):
    array = np.zeros(sizeOfArray)
    sampledata[h5datasetName].read_direct(array)
    np.savetxt(csvFilename, array, delimiter = ',')

def writeCCADataToCSV(sampledata, numOfGrains, stepcount, datasets, steps, filename):
    for step in steps:
        writedata = np.arange(1, numOfGrains + 1)
        header = 'GrainIDs'
        for dataset in datasets:
            header = header + ',' + dataset
            dataArr = np.zeros((numOfGrains, stepcount))
            sampledata[dataset].read_direct(dataArr)
            writedata = np.vstack((writedata, dataArr[:,step]))

        writedata = np.transpose(writedata)
        np.savetxt(filename + 'Step' + str(step) + '.csv', writedata, delimiter = ',', header = header, comments='')

def writeDataToCSV(filename):
    sampledata = readHDF5(filename, 'r+')
    stepcount = 0
    grainIDs = retrieveGrainIDs(sampledata)
    numOfGrains = int(np.nanmax(grainIDs))

    for step in sampledata:
        if ('Step-' in step):
            stepcount += 1

    datasetNames = [['MeanSVM', 'MeanEVM', 'sigmaSVM', 'sigmaEVM', 'maxSVM', 'maxEVM', 'minSVM', 'minEVM', 'medianSVM', 'medianEVM', 'grainVolume', 'avgSlipSys', 'grainAvgphi1', 'grainAvgPhi', 'grainAvgphi2', 'surfaceAreaVolumeRatio', 'grainAvgQuat0', 'grainAvgQuat1', 'grainAvgQuat2', 'grainAvgQuat3'],
    ['StepAverages', 'AllPoints', 'MeanBCCAvgs', 'MeanHCPAvgs']]

    for name in [item for sublist in datasetNames for item in sublist]:
        if (name not in sampledata):
            writeDatasetToHDF5(filename)

    topname = filename.split('.')[0]

    fileNames = [['GrainMeanSVM', 'GrainMeanEVM', 'GrainSigmaSVM', 'GrainSigmaEVM', 'GrainMaxSVM', 'GrainMaxEVM', 'GrainMinSVM', 'GrainMinEVM', 'GrainMedianSVM', 'GrainMedianEVM', 'GrainVolume', 'SlipSystems', 'Phi1Angle', 'PhiAngle', 'Phi2Angle', 'SurfaceAreaVolumeRatio', 'GrainAvgQuat0', 'GrainAvgQuat1', 'GrainAvgQuat2', 'GrainAvgQuat3'],
    ['TimeStepGrainAvg', 'TimeStepVolumeAvg', 'TimeBCCAvg', 'TimeHCPAvg']]

    for index in range(2):
        for datastring_compare(stringset in range(len(datasetNames[index])):
            if (index == 0):
                arrshape = (numOfGrains, stepcount)
            elif (index == 1):
                arrshape = (stepcount, 2)

            writeDatasetToCSV(sampledata, datasetNames[index][dataset], arrshape, topname + fileNames[index][dataset] + '.csv')

    writeCCADataToCSV(sampledata, numOfGrains, stepcount, ['grainVolume', 'surfaceAreaVolumeRatio', 'MeanSVM', 'MeanEVM'], [0,1,2,3,4,5,6,7,8,9], topname + 'CCA')


for vol in ['f20_', 'f40_', 'f60_']:
    for datatype in ['eqdata.h5', 'diskdata.h5', '1051data.h5']:
        writeDataToCSV(vol + datatype)
