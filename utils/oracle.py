
from cProfile import label
import numpy as np
import torch
import os

from utils.general import xywh2xyxy
from utils.metrics import box_iou
from utils.acquisition import loadFile

from utils.al_helpers import annotate_image

#load acq and labels check iou


#TODO: Test 
#load GroundTruth Box corresponding to selected Image
def loadLabels(labelPath, files=None):

    labels = []   

    if(files is None):
        files = []
        fileList = os.listdir(labelPath)
        for file in fileList:
            split = file.split(".")
            if(split[1] == "txt"):
                files.append(split[0])
        
 
    for file in files:
        labelFile = labelPath + "/" + file + ".txt"
        if os.path.exists(labelFile):
            with open(labelFile) as f:
                label = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels [class, x,y,w,h]
        else:
            label = []    

        if(len(label)):
            
            classes = torch.from_numpy(label[:,0])
            boxes = torch.from_numpy(label[:,1:])
            boxes = xywh2xyxy(boxes)
            
            labels.append(torch.cat((boxes, classes[:,None]), -1))                  # labels [x,y,x,y, class]
        else:
            
            labels.append(torch.empty(0))
        
       

    #returns tensor with box labels; shape [x,y,5] [images, labels, boxes] box => [x,y,x,y, class]
    return labels, files



#compare GT to prediciton and decide for Supervision Level (Pseudo, low supervision, high supervision )
def compare(gt, pred, name):

    if(not len(gt)):
        return []

    hits = torch.zeros(len(gt))
    
    ious = box_iou(pred[:,:4], gt[:,:4])

    for i, iou in enumerate(ious):
        maxIoU = torch.max(iou)
        index = (iou == maxIoU).nonzero(as_tuple=True)

        #check if classes match
        if (pred[i,4] == gt[index,4]):
            hits[index] = maxIoU
        else:
            #TODO Handle by setting index to 0 and repeat step
            print("Wrong class")
  
    return hits


def calcLabelingTime(hits):


    h = 0
    ph = 0
    miss = 0

    labelingTime = 7.8  #7.8 seconds base time for every image

         #TODO check timing and quote here
    for hit in hits:
        if float(hit) > 0.90: # Label hit 
            h += 1
            labelingTime += 1.6             # Verifing the label by clicking yes (arXiv:1602.08405v3)
        elif float(hit) > 0.8: #label partly hit and in need of correction
            ph += 1
            labelingTime += 1.6 + 42             #TODO need timing for bb correction      
        else:       #label not hit
            miss += 1
            labelingTime += 1.6 + 42              # Redrawing box with quality control (arXiv:1602.08405v3)

    print ("Hits: " + str(h) + " - " + "Part Hits: " + str(ph) + " - " + "Misses: " + str(miss))

    return labelingTime, h, ph, miss





#Three Modes
    #1. groundTruth only:   Calculate standard labeling time
    #2. gt + acq:           Calculate labeling time of selected data
    #3. Full:               Calculate labeling time as if a human is given the machine labels and decides of label is hit or not 
def autOracle(gtPath, acqPath=None, predPath=None):

    labelingTimeTotal, hTotal, phTotal, missTotal = 0,0,0,0


    #get Acquisition Names
    if(acqPath is not None):
        acq = loadFile(acqPath + "/selection.txt")
        fileNames = [a[0] for a in acq] # only names not acq value

        gtLabels, _ = loadLabels(gtPath + "/labels", files=fileNames)
    else:
        # load all label files from gtPath
        gtLabels, fileNames = loadLabels(gtPath + "/labels")

    #load ground truth labels
    
    #load predicted labels
    if(predPath is not None):
        predLabels, _ = loadLabels(predPath + "/labels", files=fileNames)
        

    print(len(gtLabels))
    print(len(predLabels))
    #for each gtBB in gtBBs
    for i, name in enumerate(fileNames):
        
        if(predPath is None):
            #annotate all images with full time
            hits = torch.zeros(len(gtLabels[i]))
        else:
            hits = compare(gtLabels[i], predLabels[i], name)

        labelingTime, h, ph, miss = calcLabelingTime(hits)

        labelingTimeTotal += labelingTime
        hTotal += h
        phTotal += ph
        missTotal += miss
        
        
        #imagePath = gtPath+"/images/"+name+".jpg"
        #annoPath = gtPath+"/images/anno"+name+".jpg"
        #if(predPath is not None):
        #    annotate_image(imagePath,annoPath,gtLabels[i], predLabels[i], hits)
        

    # compare gtBB to all predBB and check if there is an IOU > Threshold (~95-99)
    # if yes calc time for yes
    # if no check if there was an hit at all
    # hit yes but bad => calc time to correct
    # hit no => calc time to redraw

    print("Total:")
    print ("Hits: " + str(hTotal) + " - " + "Part Hits: " + str(phTotal) + " - " + "Misses: " + str(missTotal))

    return labelingTimeTotal