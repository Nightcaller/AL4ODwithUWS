
from cProfile import label
import numpy as np
import torch
import os

print(os.getcwd())

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import box_iou
from utils.acquisition import loadFile


from utils.al_helpers import annotate_image
from utils.al_helpers import save_text

#load acq and labels check iou



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

    if(not len(gt) or not len(pred)):
        return [], []

    hits = torch.zeros(len(gt))
    indices = [-1] * len(gt)


    ious = box_iou(pred[:,:4], gt[:,:4])


    for i, iou in enumerate(ious):
        maxIoU = torch.max(iou)
        gtIndex = (iou == maxIoU).nonzero(as_tuple=False)[0]

       
        #check if classes match
        if (pred[i,4] == gt[gtIndex,4] or maxIoU == 0):
            hits[gtIndex] = maxIoU
            indices[gtIndex] = i
        else:
            #TODO Handle by setting index to 0 and repeat step
            print("Wrong class")
  
    return hits, indices


def calcLabelingTime(hits):


    h = 0
    ph = 0
    miss = 0
    draw = 0

    labelingTime = 7.8  #7.8 seconds base time for every image

    #TODO check timing and quote here
    for hit in hits:
        if float(hit) > 0.90: # Label hit 
            h += 1
            labelingTime += 1.6             # Verifing the label by clicking yes (arXiv:1602.08405v3)
        elif float(hit) > 0.8: #label partly hit and in need of correction
            ph += 1
            labelingTime += 1.6 + 42             #TODO need timing for bb correction      
        elif float(hit) == 0:
            draw += 1
            labelingTime += 42
        else:       #label not hit
            miss += 1
            labelingTime += 1.6 + 42        # Redrawing box with quality control (arXiv:1602.08405v3)

    #print ("Hits: " + str(h) + " - " + "Part Hits: " + str(ph) + " - " + "Misses: " + str(miss))

    return labelingTime, h, ph, miss, draw


def create_pseudo_label(hits,indices, gt, pred, threshold):

    for i, hit in enumerate(hits):
        if hit >= threshold:
            gt[i] = pred[indices[i]]

    return gt

def save_pseudo_label(gtPath, fileName, gtLabels):
    print("!!!!!!!!!!!!!!!!!!")
    print(len(gtLabels))
    gtLabels = xyxy2xywh(gtLabels[:-1])
    print("###########")
    print(len(gtLabels))
    with open(gtPath + "/" + fileName + '.txt', 'w') as f:

        for label in gtLabels:
            label = label.numpy().tolist()
            label = label[-1:] + label[:-1] 
            f.write(('%g ' * len(label)).rstrip() % tuple(label) + '\n')

    return 0



#Three Modes
    #1. groundTruth only:   Calculate standard labeling time
    #2. gt + acq:           Calculate labeling time of selected data
    #3. Full:               Calculate labeling time as if a human is given the machine labels and decides of label is hit or not 
def autOracle(gtPath, savePath= None,  acqPath=None, predPath=None, cycle="0"):

    labelingTimeSSL,labelingTimeAL, hTotal, phTotal, missTotal, drawTotal = 0,0,0,0,0,0
    pseudo_label_threshold = 0.9

    #get Acquisition Names
    if(acqPath is not None):
        acq = loadFile(acqPath + "/selection.txt")
        fileNames = [a[0] for a in acq] # only names not acq value
        acqType = [a[2] for a in acq]

        gtLabels, _ = loadLabels(gtPath + "/labels", files=fileNames)
    else:   #calc pure labeling time for all gtPath Labels
        # load all label files from gtPath
        gtLabels, fileNames = loadLabels(gtPath + "/labels")

    #load predicted labels
    if(predPath is not None):
        predLabels, _ = loadLabels(predPath + "/labels", files=fileNames)
        
    #for each gtBB in gtBBs
    for i, name in enumerate(fileNames):
        
        if(predPath is None) or acqType[i] == "al":
            #annotate all images with full time
            hits = torch.zeros(len(gtLabels[i]))
        else:
            hits, indices = compare(gtLabels[i], predLabels[i], name)
            if len(hits):
                if acqType[i] == "ssl" and max(hits) >= pseudo_label_threshold:
                    gtLabels[i] = create_pseudo_label(hits, indices,gtLabels[i], predLabels[i], pseudo_label_threshold)
                    save_pseudo_label(gtPath, name, gtLabels[i])

        labelingTime, h, ph, miss, draw = calcLabelingTime(hits)

        if(predPath is not None):
            if acqType[i] == "al":
                labelingTimeAL += labelingTime
            else:
                labelingTimeSSL += labelingTime
        else:
            labelingTimeAL += labelingTime


        hTotal += h
        phTotal += ph
        missTotal += miss
        drawTotal += draw
        
        
        #imagePath = gtPath+"/images/"+name+".jpg"
        #annoPath = gtPath+"/images/anno"+name+".jpg"
        #if(predPath is not None):
        #    annotate_image(imagePath,annoPath,gtLabels[i], predLabels[i], hits)
        

    # compare gtBB to all predBB and check if there is an IOU > Threshold (~95-99)
    # if yes calc time for yes
    # if no check if there was an hit at all
    # hit yes but bad => calc time to correct
    # hit no => calc time to redraw
    labelingTimeTotal = labelingTimeAL + labelingTimeSSL
    if not (os.path.exists(savePath)):
        os.makedirs(savePath)

    #("Run,","AL Time in Hours, SSL Time in Hours, Hits, Part Hits, Misses, NewDraws"), 
    saveText = [
        (str(cycle[-1]) + " ," ,str(labelingTimeAL / 3600) + "," + str(labelingTimeSSL / 3600) + "," + str(hTotal) + ", " + str(phTotal)+  ", " + str(missTotal) + ", " + str(drawTotal) )
    ]

    save_text(saveText, savePath, "oracle")

    print("Total Labeling Time: " + str(labelingTimeTotal / 3600) + " Hours")
    print("AL Labeling Time: " + str(labelingTimeAL / 3600) + " Hours")
    print("SSL Labeling Time: " + str(labelingTimeSSL / 3600) + " Hours")
    print("Hits: " + str(hTotal) + " - " + "Part Hits: " + str(phTotal) + " - " + "Misses: " + str(missTotal) + " - " + "New Draws: " + str(drawTotal) )

    return labelingTimeTotal



if __name__ == "__main__":
    
    gtPath = "/Users/mhpinnovation/Documents/Daniel/Master/datasets/AcqTest"
    predPath = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp473"
    acqFile = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp473/acquisition"
    autOracle(gtPath, gtPath, acqFile, predPath, "")