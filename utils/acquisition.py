import os


def loadAcqFile(source):

    file = open(source, 'r')
    lines = file.readlines()
    acq = []

    for line in lines:
        
        acq.append(line.split())

    return acq


#move selected Images from paths to target directory (trainingSet)
def moveSelection(imageNames, sourceDir, targetDir):

    for name in imageNames:
        os.rename(sourceDir + "/images/" +  name + ".jpg", targetDir + "/images/" + name + ".jpg")
        os.rename(sourceDir + "/labels/" +  name + ".txt", targetDir + "/labels/" + name + ".txt")
        

    return True





def acquisition(acqSource, source, target, threshold, modes=["bot"]):

    acq = loadAcqFile(acqSource)

    selection = []
    acqN = len(acq)
    quantil = int(acqN * threshold)

    
    if any("top" in s for s in modes):
        #TODO: Filter out zeros in acq cause those images are empty and not interesting at all
        selection = acq[:quantil]
    elif any("bot" in s for s in modes):
        selection = acq[acqN-quantil:]
    

    #return only the names
    selection = [a[0] for a in selection]


    success = moveSelection(selection, source, target)

    return success





