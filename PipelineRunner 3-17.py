import json
import os
import sys
import math
import glob
import numpy as np
import random
import csv
import subprocess
import time


#Before using, open Dream3D and set the folder that you want the output files in.
#Functions here used to change the pipeline only affect the name of the output file
#not the directory.

#Type directory containing json files for Dream3D Pipelines
pipelineDirectory = '/home/jackyl/Desktop/Dream3DPyTest/Pipes/BasicPipeline'

#PipelineRunnerDirectory
# Currently this requires that the PipelineRunner file be placed in the Plugins
# directory of the DREAM3D files.
pipeRunnerDirectory = '/home/jackyl/Desktop/Dream3DPyTest/Dream3D-6.3.29/Plugins'

#Path to output directory
outputDirectory = '/home/jackyl/Desktop/Dream3DPyTest/3-17Baseline'



################################################
#Housekeeping - Managing files
################################################

def openPipeline(filePath):
    #Open JSON for editing
    with open(filePath, 'r') as jsonData:
        pipeData = json.load(jsonData)
        return pipeData


def updatePipeline(pipeData, filePath):
    #Overwrite JSON
    with open(filePath, "w") as jsonFile:
        jsonFile.write(json.dumps(pipeData))


def runPipelineRunner(pipeline):
    # Changed Working Directory to where my pipelinerunner command was
    # This may not be necessary on your machine, check with PipelineRunner Docs for Dream3D
    # and adjust cwd as necessary
    # Runs PipelineRunner in Terminal - subprocess should not continue unless previous is done.
     subprocess.call(['./PipelineRunner', '-p', pipeline],
        cwd =pipeRunnerDirectory)
    #
    # This is also valid, and allows starting several DREAM3D processes, but does not stop
    # even if it uses all the RAM available and crashes
    # USE AS YOUR OWN RISK (Add a time.sleep call to the trial function)
    # subprocess.Popen(['./PipelineRunner', '-p', pipeline],
    #    cwd=pipeRunnerDirectory)







################################################
# JSON Editing Functions
################################################

def changeMuAndSD(pipeData, newMu, newSD, phase=1, cutoff=4):
    #Overwrite JSON with new Mu and SD
    for item in pipeData:
        if (item != "PipelineBuilder" and int(item) == 0):
            section = item
    pipeData[section]['StatsDataArray'][str(phase)]['FeatureSize Distribution']['Average'] = newMu
    pipeData[section]['StatsDataArray'][str(phase)]['FeatureSize Distribution']['Standard Deviation'] = newSD
    pipeData[section]['StatsDataArray'][str(phase)]['Feature_Diameter_Info'][1] = math.exp(newMu + newSD*cutoff)
    pipeData[section]['StatsDataArray'][str(phase)]['Feature_Diameter_Info'][2] = math.exp(newMu - newSD*cutoff)

def changePhaseFraction(pipeData, fraction, phase=1):
    #Overwrite JSON with new volume fraction for the phase
    for item in pipeData:
        if (item != "PipelineBuilder" and int(item) == 0):
            section = item
    pipeData[section]['StatsDataArray'][str(phase)]['PhaseFraction'] = fraction

def changeDimensions(pipeData, inputX, inputY, inputZ):
    #Overwrite JSON with new Volume Size
    pipeData['01']['Dimensions']['y'] = inputY
    pipeData['01']['Dimensions']['x'] = inputX
    pipeData['01']['Dimensions']['z'] = inputZ

def changeResolution(pipeData, inputX, inputY, inputZ):
    #Overwrite JSON with new Resolution
    pipeData['01']['Resolution']['y'] = inputY
    pipeData['01']['Resolution']['x'] = inputX
    pipeData['01']['Resolution']['z'] = inputZ

def changeShapeDist(pipeData, alpha1, beta1, alpha2, beta2, phase=1):
    #Overwrite JSON with new shape distributions (Controlling Alpha/Beta parameters)
    for item in pipeData:
        if (item != "PipelineBuilder" and int(item) == 0):
            section = item
    pipeData[section]['StatsDataArray'][str(phase)]['FeatureSize Vs B Over A Distributions']['Alpha'] = [alpha1]
    pipeData[section]['StatsDataArray'][str(phase)]['FeatureSize Vs B Over A Distributions']['Beta'] = [beta1]
    pipeData[section]['StatsDataArray'][str(phase)]['FeatureSize Vs C Over A Distributions']['Alpha'] = [alpha2]
    pipeData[section]['StatsDataArray'][str(phase)]['FeatureSize Vs C Over A Distributions']['Beta'] = [beta2]

def changeOutputFileName(pipeData, typeOfFile, newFileName, outputDir=outputDirectory):
    # NOTE - Only changes the file name, does not change containing directories
    # DO NOT HAVE "/" IN THE NEW FILE NAME
    # typeOfFile - csv, dream3d - depends if the filter exists already
    if (typeOfFile == "csv"):
        for part in pipeData:
            if (pipeData[part].get('Filter_Human_Label', 0) == 'Write Feature Data as CSV File'):
                section = part
        output = 'FeatureDataFile'
    elif (typeOfFile == "dream3d"):
        for part in pipeData:
            if (pipeData[part].get('Filter_Human_Label', 0) == 'Write DREAM.3D Data File'):
                section = part
        output = 'OutputFile'
    elif (typeOfFile == 'polefig'):
        for part in pipeData:
            if (pipeData[part].get('Filter_Human_Label', 0) == 'Write Pole Figure Images'):
                section = part
    elif(typeOfFile == 'FFT'):
        for part in pipeData:
            if (pipeData[part].get('Filter_Human_Label', 0) == "Write Los Alamos FFT File"):
                section = part

    if (outputDir != None and typeOfFile != 'polefig' and typeOfFile != 'FFT'):
        pipeData[section][output] = outputDir + "/" + newFileName
    elif (typeOfFile == 'polefig'):
        pipeData[section]['OutputPath'] = outputDir
        pipeData[section]['ImagePrefix'] = newFileName
    elif (typeOfFile == 'FFT'):
        pipeData[section]['OutputFile'] = outputDir + "/" + newFileName
        pipeData[section]['FeatureIdsArrayPath']['OutputFile'] = outputDir + "/" + newFileName
    else:
        curName = pipeData[section][output]
        partList = curName.split("/")
        partList[-1] = newFileName
        newName = '/'.join(partList)
        pipeData[section][output] = newName

def changeODF(pipeData, e1, e2, e3, wt, sigma, phase=1):
    #Change ODF requires e1, e2, e3 to be in degrees
    if (type(e1) != list):
        e1 = [e1]
    if (type(e2) != list):
        e2 = [e2]
    if (type(e3) != list):
        e3 = [e3]
    if (type(wt) != list):
        wt = [wt]
    if (type(sigma) != list):
        sigma = [sigma]

    e1 = list(map(lambda x: math.radians(x), e1))
    e2 = list(map(lambda x: math.radians(x), e2))
    e3 = list(map(lambda x: math.radians(x), e3))

    if (e1 == [] or e2 == [] or e3 == [] or wt == [] or sigma == []):
        pipeData['00']['StatsDataArray'][str(phase)]['ODF-Weights'] = {}
    else:
        pipeData['00']['StatsDataArray'][str(phase)]['ODF-Weights']['Weight'] = wt
        pipeData['00']['StatsDataArray'][str(phase)]['ODF-Weights']['Sigma'] = sigma
        pipeData['00']['StatsDataArray'][str(phase)]['ODF-Weights']['Euler 1'] = e1
        pipeData['00']['StatsDataArray'][str(phase)]['ODF-Weights']['Euler 2'] = e2
        pipeData['00']['StatsDataArray'][str(phase)]['ODF-Weights']['Euler 3'] = e3






################################################
# Texture Helper Functions
################################################

def eulerAnglesToMatrix(eulerAngle):
    #Angles are in Degrees
    phi1 = eulerAngle[0]
    Phi = eulerAngle[1]
    phi2 = eulerAngle[2]

    Z1 = np.matrix([[math.cos(phi1), math.sin(phi1), 0],
                    [-math.sin(phi1), math.cos(phi1), 0],
                    [0, 0, 1]])

    Z2 = np.matrix([[math.cos(phi2), math.sin(phi2), 0],
                    [-math.sin(phi2), math.cos(phi2), 0],
                    [0, 0, 1]])

    X = np.matrix([[1, 0, 0],
                    [0, math.cos(Phi), math.sin(Phi)],
                    [0, -math.sin(Phi), math.cos(Phi)]])

    mat = Z2 * X * Z1
    return mat

def matrixToEuler(g):
    if (g[2,2] == 1):
        A2 = 0
        A1 = np.arctan2(g[0,1], g[0,0])/2
        A3 = A1
    else:
        A2 = math.acos(g[2, 2])
        A1 = np.arctan2(g[2,0]/math.sin(A2), -g[2,1]/math.sin(A2))
        A3 = np.arctan2(g[0,2]/math.sin(A2), g[1,2]/math.sin(A2))
    return np.degrees(np.matrix([A1, A2, A3]))

def millerIndexToMatrix(b, n):
    #Requires b and n to be np.matrix types
    bnorm = b / np.linalg.norm(b)
    nnorm = n / np.linalg.norm(n)
    t = np.cross(nnorm, bnorm)
    tnorm = t / np.linalg.norm(t)
    g = np.hstack((np.matrix.transpose(bnorm),
        np.matrix.transpose(tnorm), np.matrix.transpose(nnorm)))
    return g

def removeOuterList(variantList):
    endList = []
    for mat in variantList:
        for elem in mat:
            endList.append(elem)
    return endList



def generate12Variants(BCC_euler):
    variantList = []
    g0 = eulerAnglesToMatrix(BCC_euler)
    n = np.matrix([ [1, 1, 0],
                    [1, 1, 0],
                    [1, -1, 0],
                    [1, -1, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, -1],
                    [0, 1, -1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

    b = np.matrix( [[-1, 1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [1, 1, 1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 1],
                    [1, 1, 1]])

    for row in range(12):
        gBCC = millerIndexToMatrix(b[row,:], n[row,:])
        delG = np.linalg.inv(gBCC)
        gHCP = delG*g0
        variant = matrixToEuler(gHCP)
        variant = variant
        variantList.append(variant.tolist())
    variantList = removeOuterList(variantList)
    return variantList

# Pipeline Trial Helpers

def normalize(pipeData):
    changeDimensions(pipeData, 128, 128, 128)
    changeResolution(pipeData, 1, 1, 1)
    changeMuAndSD(pipeData, 10, 0.2, 1, 2)
    changeMuAndSD(pipeData, 2, 0.1, 2, 2)
    changePhaseFraction(pipeData, 0.8, 1)
    changePhaseFraction(pipeData, 0.2, 2)
    changeODF(pipeData, [], [], [], [], [], 1)
    changeODF(pipeData, [], [], [] ,[], [], 2)
    changeShapeDist(pipeData, 15.3539, 1.70799, 15.0549, 1.40672, 1)
    changeShapeDist(pipeData, 15.3539, 1.70799, 15.0549, 1.40672, 2)
    changeOutputFileName(pipeData, 'csv', 'Test.csv')
    changeOutputFileName(pipeData, 'dream3d', 'Test.dream3d')
    #changeOutputFileName(pipeData, 'FFT', 'Test.txt')

def normalizePipeline():
    for pipeline in glob.glob(pipelineDirectory + '/*'):
        pipeData = openPipeline(pipeline)
        normalize(pipeData)
        updatePipeline(pipeData, pipeline)


def runMatrixTrial():
    for pipeline in glob.glob(pipelineDirectory + '/*.json'):
        variantList = generate12Variants([0,0,0])
        for ODF in ['V1611', 'V1611279', 'VAll']:
            for trial in range(16, 21):
                pipeData = openPipeline(pipeline)
                normalize(pipeData)

                if (ODF == 'V1611' and trial in [16, 17, 18]):
                    continue

                e1 = []
                e2 = []
                e3 = []
                wt = []
                sig = []

                if (ODF == 'V1'):
                    variants = [1]
                elif (ODF == 'V1611'):
                    variants = [1, 6, 11]
                elif (ODF == 'V1611279'):
                    variants = [1, 6, 11, 2, 7, 9]
                elif (ODF == 'VAll'):
                    variants = [1,2,3,4,5,6,7,8,9,10,11,12]

                for var in variants:
                    e1.append(variantList[var - 1][0])
                    e2.append(variantList[var - 1][1])
                    e3.append(variantList[var - 1][2])
                    wt.append(50000)
                    sig.append(1)

                changeODF(pipeData, 0, 0, 0, 50000, 1, 1)
                changeODF(pipeData, e1, e2, e3, wt, sig, 2)

                filename = 'Tex ' + ODF + ' T' + str(trial)
                changeOutputFileName(pipeData, 'dream3d', filename + '.dream3d', outputDirectory + '/Dream3Ds')
                changeOutputFileName(pipeData, 'csv', filename + '.csv', outputDirectory + '/CSVs')
                updatePipeline(pipeData, pipeline)
                runPipelineRunner('\ '.join(pipeline.split(' ')))


#runMatrixTrial()

######################################################################
# EXAMPLE TRIAL GENERATING FUNCTIONS
######################################################################

def runTrials():
    for pipeline in glob.glob(pipelineDirectory + '/*'):
        for mu in [4,3,2,1]:
            for trial in range(1, 21):
                pipeData = openPipeline(pipeline)
                normalize(pipeData)

                filename = 'Baseline ' + str(mu) + ' T' + str(trial)

                changeMuAndSD(pipeData, mu, 0.1, 2)

                changeShapeDist(pipeData, 30.6667, 1.40255, 4.02051, 27.1268, 2)

                changeOutputFileName(pipeData, 'dream3d', filename + '.dream3d', outputDirectory + '/Dream3Ds')
                changeOutputFileName(pipeData, 'csv', filename + '.csv', outputDirectory + '/CSVs')
                updatePipeline(pipeData, pipeline)
                runPipelineRunner('\ '.join(pipeline.split(' ')))

runTrials()

def monteCarloTrials(pipeline):
    ShapeList = []
    Alpha0List = []
    R10101 = [30.8667, 1.94892, 4.62427, 27.8317]
    R1081 = [24.768,7.31987,4.08771,27,647]
    R1061 = [18.4559, 3.0034, 4.26084, 27.9098]
    R1041 = [13.248, 18.5636, 4.61473, 27.1804]
    R1021 = [7.48646, 24.5499, 4.29031, 27.5526]
    R1011 = [4.9762, 27.392, 4.56662, 27.266]
    R881 = [30.3399,1.51865,5.44843,26.6252]
    R861 = [23.2008,8.94827,5.29842,26.8199]
    R841 = [16.5193,15.8217,5.54732,27.3636]
    R821 = [8.85877,23.1557,4.89909,27.0291]
    R811 = [5.24212,27.2869,4.71643,27.0683]
    R661 = [30.0046,1.29145,6.87025,25.828]
    R641 = [20.5344,11.1928,6.09489,25.2599]
    R621 = [10.8553,20.854,6.42851,26.1665]
    R611 = [6.23031,25.8903,5.93231,25.7963]
    R441 = [30.9292,1.73645,8.56733,22.8776]
    R421 = [16.0893,16.1096,8.71889,23.0372]
    R411 = [8.53004,22.9359,9.04091,23.7277]
    R221 = [30.5024,1.39213,16.1871,15.8251]
    R211 = [15.7156,16.1262,16.5204,15.8496]
    R111 = [15.3539, 1.70799, 15.0549, 1.40672]


    for roll in [R10101, R1081, R1061, R1041, R1021, R1011,
                    R881, R861, R841, R821, R811,
                    R661, R641, R621, R611,
                    R441, R421, R411,
                    R221, R211, R111]:
        Alpha0List.append(roll[0])
        ShapeList.append(roll)

    for trial in range(1,501):
        VolFrac = random.random()*0.6
        phi1 = random.uniform(0, 360)
        Phi = random.uniform(0, 180)
        phi2 = random.uniform(0, 360)
        BCCTexture = [phi1, Phi, phi2]
        HCPVariants = generate12Variants(BCCTexture)

        BCCWeight = []
        BCCWeight.append(random.randint(0, 100000))

        HCPWeightList = []
        for weight in range(12):
            if (random.randint(0, 1) == 0):
                HCPWeightList.append(0)
            else:
                weight = random.randint(0, 100000)
                HCPWeightList.append(weight)

        E1 = []
        E2 = []
        E3 = []
        for elem in HCPVariants:
            E1.append(elem[0])
            E2.append(elem[1])
            E3.append(elem[2])

        HCPMu = random.uniform(1, 2)
        HCPSigma = random.uniform(0.1, 1)

        ShapePara = ShapeList[random.randint(0,20)]
        shapeIndex = Alpha0List.index(ShapePara[0])
        if (shapeIndex == 0):
            tag = 'R10101'
        elif (shapeIndex == 1):
            tag = 'R1081'
        elif (shapeIndex == 2):
            tag = 'R1061'
        elif (shapeIndex == 3):
            tag = 'R1041'
        elif (shapeIndex == 4):
            tag = 'R1021'
        elif (shapeIndex == 5):
            tag = 'R1011'
        elif (shapeIndex == 6):
            tag = 'R881'
        elif (shapeIndex == 7):
            tag = 'R861'
        elif (shapeIndex == 8):
            tag = 'R841'
        elif (shapeIndex == 9):
            tag = 'R821'
        elif (shapeIndex == 10):
            tag = 'R811'
        elif (shapeIndex == 11):
            tag = 'R661'
        elif (shapeIndex == 12):
            tag = 'R641'
        elif (shapeIndex == 13):
            tag = 'R621'
        elif (shapeIndex == 14):
            tag = 'R611'
        elif (shapeIndex == 15):
            tag = 'R441'
        elif (shapeIndex == 16):
            tag = 'R421'
        elif (shapeIndex == 17):
            tag = 'R411'
        elif (shapeIndex == 18):
            tag = 'R221'
        elif (shapeIndex == 19):
            tag = 'R211'
        elif (shapeIndex == 20):
            tag = 'R111'


        pipeData = openPipeline(pipeline)
        normalize(pipeData)

        changePhaseFraction(pipeData, VolFrac, 2)
        changePhaseFraction(pipeData, 1 - VolFrac, 1)

        changeMuAndSD(pipeData, HCPMu, HCPSigma, 2)

        changeODF(pipeData, BCCTexture[0], BCCTexture[1], BCCTexture[2], BCCWeight, 1, 1)
        changeODF(pipeData, E1, E2, E3, HCPWeightList, [1]*12, 2)

        changeShapeDist(pipeData, ShapePara[0], ShapePara[1], ShapePara[2], ShapePara[3], 2)

        filename = 'Random' + ' T' + str(trial)
        writename = outputDirectory + '/Trial Parameters.csv'

        changeOutputFileName(pipeData, 'dream3d', filename + '.dream3d', outputDirectory + '/Dream3Ds')
        changeOutputFileName(pipeData, 'csv', filename + '.csv', outputDirectory + '/CSVs')
        updatePipeline(pipeData, pipeline)

        runPipelineRunner('\ '.join(pipeline.split(' ')))
        time.sleep(25)

        with open(writename, 'a') as csvfile:
            parfile = csv.writer(csvfile, delimiter = ',')
            parfile.writerow([trial, HCPMu, HCPSigma, phi1, Phi, phi2,
                                ShapePara[0], ShapePara[1], ShapePara[2],
                                ShapePara[3], VolFrac, tag, HCPWeightList[0],
                                HCPWeightList[1],HCPWeightList[2],HCPWeightList[3],
                                HCPWeightList[4],HCPWeightList[5],HCPWeightList[6],
                                HCPWeightList[7],HCPWeightList[8],HCPWeightList[9],
                                HCPWeightList[10],HCPWeightList[11], BCCWeight[0]])



##########################
# Template FUNCTION
##########################

def runTrials():
    for pipeline in glob.glob(pipelineDirectory + '/PercPipe2Filters.json'):
        #Add additional for loops to run several trials of several parameter ranges
        for volFrac in [0.2, 0.4, 0.6]:
            for shape in ['111', '10101', '1051', '551', '1011']:
                for trial in range(1):

                #Read current json file-
                    pipeData = openPipeline(pipeline)

                #Clear the pipeline (OPTIONAL, but Recommended)
                    normalize(pipeData)

                #Change File Name as needed
                    # Volfrac of 0.88 results in BCC= 0.78, HCP = 0.21
                    # Volfrac of 0.875 results in BCC = 0.775, HCP = 0.225
                    # Volfrac of 0.85 results in BCC = 0.73, HCP = 0.27
                    # Volfrac of 0.8 results in BCC = 0.66, HCP = 0.34
                    # Volfrac of 0.75 results in BCC = 0.6, HCP = 0.4 <--
                    # Volfrac of 0.7 results in BCC = 0.52, HCP =0.48
                    # Volfrac of 0.6 results in BCC = 0.4, HCP = 0.6 <---
                    # Volfrac of 0.4 results in BCC = 0.25, HCP = 0.75
                    # Volfrac of 0.2 results in BCC = 0.2, HCP = 0.8 <---

                    if (volFrac == 0.2):
                        volFracTag = '02'
                    elif (volFrac == 0.4):
                        volFracTag = '04'
                    elif (volFrac == 0.6):
                        volFracTag = '06'

                    if (shape == '10101'):
                        shapeParam = (30.6667, 1.40255, 4.02051, 27.1268)
                    elif (shape == '1051'):
                        shapeParam = (15.7045, 16.2173, 4.97866, 27.9236)
                    elif (shape == '551'):
                        shapeParam = (30.7033, 1.42665, 6.9861, 24.71343)
                    elif (shape == '1011'):
                        shapeParam = (4.27507, 28.0289, 4.36924, 27.8468)
                    elif (shape == '111'):
                        shapeParam = (15.3539, 1.70799, 15.0549, 1.40672)


                    changeShapeDist(pipeData, shapeParam[0], shapeParam[1], shapeParam[2], shapeParam[3], 2)
                    changePhaseFraction(pipeData, 1-volFrac , 1)
                    changePhaseFraction(pipeData, volFrac, 2)


                    filename = 'asp' + shape + "_vol" + volFracTag + "_" + str(trial) + 'Runner'
                    changeOutputFileName(pipeData, 'dream3d', filename + '.dream3d', outputDirectory + '/Dream3Ds')
                    changeOutputFileName(pipeData, 'csv', filename + '.csv', outputDirectory + '/CSVs')
                    changeOutputFileName(pipeData, 'FFT', filename + '.txt', outputDirectory + '/FFTs')

                #Change Parameters

                #Save changes to json
                    updatePipeline(pipeData, pipeline)

                #Run Terminal Command
                    runPipelineRunner('\ '.join(pipeline.split(' ')))
                    time.sleep(20)



def runTrialsFFTTestCase():
    for pipeline in glob.glob(pipelineDirectory + '/PercPipe2Filters.json'):
        for shape in [1, 2]:
            for Mu in [2, 3]:
                for volFrac in [0.2, 0.4, 0.6]:
                    pipeData = openPipeline(pipeline)
                    normalize(pipeData)

                    if (shape == 1):
                        shapeParam = (30.6667, 1.40255, 4.02051, 27.1268)
                    else:
                        shapeParam = (15.3539, 1.70799, 15.0549, 1.40672)

                    changeShapeDist(pipeData, shapeParam[0], shapeParam[1], shapeParam[2], shapeParam[3], 2)

                    changePhaseFraction(pipeData, 1 - volFrac, 1)
                    changePhaseFraction(pipeData, volFrac, 2)

                    changeMuAndSD(pipeData, Mu, 0.2, 2, 3)
                    changeMuAndSD(pipeData, 10, 0.2, 1, 4)

                    changeDimensions(pipeData, 128,128,128)
                    changeResolution(pipeData, 1,1,1)

                    if (shape == 1):
                        shapetag = '10101'
                    else:
                        shapetag = '111'

                    if (pipeline == '/home/selwyni/Desktop/Dream3DPyTest/Pipes/PercolationPipe/PercPipe1Filter.json'):
                        tag = 1
                    else:
                        tag = 2

                    filename = "10101FFT" + str(tag) + "-" + str(int(10*Mu)) + "-" + str(int(10*volFrac))

                    changeOutputFileName(pipeData, 'dream3d', filename + '.dream3d', outputDirectory + '/Dream3Ds')
                    changeOutputFileName(pipeData, 'csv', filename + '.csv', outputDirectory + '/CSVs')
                    changeOutputFileName(pipeData, 'FFT', filename + '.txt', outputDirectory + '/FFTs')

                    #Save changes to json
                    updatePipeline(pipeData, pipeline)

                    #Run Terminal Command
                    runPipelineRunner('\ '.join(pipeline.split(' ')))
                    time.sleep(45)


def runPercTrials():
    for pipeline in glob.glob(pipelineDirectory + '/PercPipe2Filters.json'):
        for trial in range(1,101):
            Mu = 2 + random.random()
            shape = random.randint(1, 5)
            if (shape == '10101'):
                shapeParam = (30.6667, 1.40255, 4.02051, 27.1268)
            elif (shape == '1051'):
                shapeParam = (15.7045, 16.2173, 4.97866, 27.9236)
            elif (shape == '551'):
                shapeParam = (30.7033, 1.42665, 6.9861, 24.71343)
            elif (shape == '1011'):
                shapeParam = (4.27507, 28.0289, 4.36924, 27.8468)
            elif (shape == '111'):
                shapeParam = (15.3539, 1.70799, 15.0549, 1.40672)
