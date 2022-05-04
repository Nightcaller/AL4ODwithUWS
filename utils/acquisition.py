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
    from bisect import bisect_left

    acq = loadFile(acqSource + "/" + acqType + ".txt")

    selection = []
    acqN = len(acq)

    print(acq) 

    valuesWithZero = [float(a[1]) for a in acq]
    values = [float(a[1]) for a in acq if float(a[1]) != 0]
    
    #TODO: Set maximum acq size
    topQuantil = values[0] * (1+threshold)
    topIndex = bisect_left(valuesWithZero, topQuantil)
    smallestValueIndex = valuesWithZero.index(values[0])
    #
    botQuantil = valuesWithZero[-1] * (1-threshold)
    botIndex = bisect_left(valuesWithZero, botQuantil)

    #quantil = int(acqN * threshold)
    print("Selection from : " + str(acqN))
    if any("top" in s for s in modes):
        
        selection = acq[smallestValueIndex:topIndex]
        print("TOP: " + str(topIndex-smallestValueIndex))

    if any("bot" in s for s in modes):
        selection += acq[botIndex:]
        print("BOT: " + str(acqN-botIndex))
    
    
    
   
    #save names in file selection.txt
    save_text(selection, acqSource, "selection")

    #return only the names
    selection = [a[0] for a in selection]

    
    

    success = moveSelection(selection, source, target)

    return success





