import os


def loadAcqFile(source):

    file = open(source, 'r')
    lines = file.readlines()
    acq = []

    for line in lines:
        
        acq.append(line.split())

    return acq


def acqSelection(acq, threshold, mode ="top"):

    selection = []
    acqN = len(acq)
    quantil = int(acqN * threshold)
    if(mode== "top"):    
        selection = acq[:quantil]
    elif(mode=="bot"):
        selection = acq[acqN-quantil:]
    

    #return only the names
    selection = [a[0] for a in selection]

    return selection



#move selected Images from paths to target directory (trainingSet)
def moveSelection(imageNames, sourceDir, targetDir):

    for name in imageNames:
        os.rename(sourceDir + "/images/" +  name + ".jpg", targetDir + "/images/" + name + ".jpg")
        os.rename(sourceDir + "/labels/" +  name + ".txt", targetDir + "/labels/" + name + ".txt")
        

    return True


