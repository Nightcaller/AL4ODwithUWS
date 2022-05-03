import os
from utils.al_helpers import save_text

def loadFile(source):

    acq = []
    file = open(source, 'r')
    lines = file.readlines()
    
    for line in lines:
        acq.append(line.split())

    file.close()
    return acq


#move selected Images from paths to target directory (trainingSet)
def moveSelection(imageNames, sourceDir, targetDir):

    try:
        for name in imageNames:
            os.rename(sourceDir + "/images/" +  name + ".jpg", targetDir + "/images/" + name + ".jpg")
            os.rename(sourceDir + "/labels/" +  name + ".txt", targetDir + "/labels/" + name + ".txt")
    except:
        return False        

    return True


#acqSource, Move From, To, How much percent, pickFrom (top,bot,mid), AL Strat (DropoutUncertainty, Random, leastConfidence)
def selection(acqSource, source, target, threshold, modes, acqType):

    acq = loadFile(acqSource + "/" + acqType + ".txt")

    selection = []
    acqN = len(acq)
    
    #topQuantil = 0
    #botQuantil = acq[-1] * (1-threshold)

    quantil = int(acqN * threshold)
   
    if any("top" in s for s in modes):
        #TODO: Filter out zeros in acq cause those images are empty and not interesting at all
        selection = acq[:quantil]
        print("top")
    elif any("bot" in s for s in modes):
        selection = acq[acqN-quantil:]
    
   
    #save names in file selection.txt
    save_text(selection, acqSource, "selection")

    #return only the names
    selection = [a[0] for a in selection]

    
    

    success = moveSelection(selection, source, target)

    return success





