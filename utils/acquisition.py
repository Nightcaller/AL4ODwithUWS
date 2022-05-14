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
def selection(acqSource, source, target, threshold, modes):
    from bisect import bisect_left, bisect_right

    acq = loadFile(acqSource + "/uncertainty.txt")

    selection = []
    acqN = len(acq)
    acqMax = int(acqN * threshold)
    
    valuesWithZero = [float(a[1]) for a in acq]
    values = [float(a[1]) for a in acq if float(a[1]) != 0]

    quantil = max(values) * threshold
   

   
    sslQuantil = values[0] + quantil 
    alQuantil = values[-1] - quantil 

    sslIndex = bisect_left(valuesWithZero, sslQuantil)
    smallestValueIndex = valuesWithZero.index(values[0])
 
    alIndex = bisect_left(valuesWithZero, alQuantil)

    print("Selection from : " + str(acqN))
    if any("ssl" in s for s in modes):
        
        if (sslIndex-smallestValueIndex > acqMax):
            sslIndex = smallestValueIndex + acqMax

        selection = acq[smallestValueIndex:sslIndex]
        print("Pseudo Labeling: " + str(sslIndex-smallestValueIndex))

    if any("al" in s for s in modes):
        if(acqN-alIndex > acqMax):
            alIndex = acqN-acqMax

        selection += acq[alIndex:]
        print("Active Learning: " + str(acqN-alIndex))
    
    
    
   
    #save names in file selection.txt
    save_text(selection, acqSource, "selection")

    #return only the names
    selection = [a[0] for a in selection]

    
    

    success = moveSelection(selection, source, target)

    return success





