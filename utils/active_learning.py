from platform import python_version_tuple
import random
from sklearn.cluster import cluster_optics_dbscan
import torch
import numpy as np

from utils.metrics import box_iou



def random_sampling(): 

    return random.uniform(0, 1)

def uncertainty(predictions, path, imgSize,mode="Entropy" , threshold_iou=0.5):
    
    objects = []
    uAll = []
    first = True
     

    #cluster all predicitions into objects 
    for prediction in predictions:
        for i, det in enumerate(prediction):

            if(len(det) == 0):
                continue

            #for *xyxy, conf, cls in reversed(det):                       # det[:,:4] => BB ; det[:,4] => Confidence, det[:,5] => Class    
             
            #for i, object in enumerate(objects):
            #iou = box_iou(object, torch.tensor(xyxy)[None,:])
            if(first):
               for d in det:
                   objects.append(d[None,:])
               first = False
               continue
            

            # enumerate all objects and check the ious of already discoverd objects
            for i, d in enumerate(det):
                ious = []
                for object in objects:
                    ious.append(torch.max(box_iou(d[None,:4], object[:,:4])))


                #assign BB to max overlap 
                maxIoU = max(ious)
                index = ious.index(maxIoU)

                if(sum(ious) == 0):   
                    objects.append(d[None,:])
                    continue
                if(d[5] == objects[index][0][5] and maxIoU >= threshold_iou):   #add to existing cluster
                    objects[index] = torch.cat((objects[index], d[None,:]), 0) 
                else:
                    objects.append(d[None,:])


    # Get the mean box from every detected object
    #   and
    # Calc uncertainty by clustering corners and returning the mean of the convex hull area (DBScan)
    for i, obj in enumerate(objects):
        
        mean = torch.mean(obj, 0)
        objects[i] = torch.cat((obj, mean[None,:]), 0) 

        if(mode == "DBScan"):
            uAll.append(cluster_dbscan(obj))
        elif(mode == "Entropy"):
            uAll.append(cluster_entropy(obj))
        elif(mode == "Margin"):
            uAll.append(cluster_margin(obj))
        elif(mode == "LC"):
            uAll.append(cluster_lc(obj))
    

    return objects, uAll


'''            for j, iou in enumerate(ious):
                
                #if there is no intersection at all it probably is a newly discoverd object
                if(torch.mean(iou) <= 0):
                    iouzbjects.append(det[j][None,:])
                    print("added new Object")
                    continue
                
                index = int(torch.argmax(iou)) % len(objects)
                objects[index] = torch.cat((objects[index], det[j][None,:]), 0) 
#                for iouXobj in iou:
#                    for j ,realIou in enumerate(iouXobj):
#                    
 #                       if(torch.mean(realIou) <= 0):
 #                           objects.append(det[j][None,:])
 #               print(iou)
            print(len(objects))

'''
        

            #if class is the same and iou is over threshhold 
            #y = torch.cat((objects[i], xyxy), 0) # concat new x
                

#Get uncertainty of cluster 
def cluster_dbscan(obj):
    
    from scipy.spatial import ConvexHull

    #send to gpu rather than cpu
    points1 = obj[:,:2].cpu().numpy()
    points2 = obj[:,2:4].cpu().numpy()

    if (len(points1) <= 2):
        return 0

    try:
        u = (ConvexHull(points1).area + ConvexHull(points2).area) / 2
    except:
        #ConvexHull function randomly throws an unknown exception
        u = 0
        pass
    
    return u
 
def cluster_entropy(obj):
    
    

    if(obj.size()[0] <=1 ):
        return 0

    cuda = torch.cuda.is_available()
    if cuda:
        obj.to('cuda:0')


    ''' 
    probs = obj[:,4].cpu().numpy()

    logProbsCPU = probs * np.log2(probs)
    numeratorCPU = 0 - sum(logProbsCPU)
    denomCPU = np.log2(probs.size)

    entropyCPU = numeratorCPU / denomCPU
    '''
    if cuda:
        size = torch.tensor(obj[:,4].size()).to('cuda:0')
    else:
        size = torch.tensor(obj[:,4].size())
    
    logProbs = torch.mul(obj[:,4] ,torch.log2(obj[:,4]))
    numerator = torch.sub(torch.tensor(0), torch.sum(logProbs))
    denom = torch.log2(size)


    entropy = torch.div(numerator, denom)
    
    

    return entropy


def cluster_margin(obj):
    u = 0

    if(obj.size()[0] <=1 ):
       return 0

    topTwo = torch.topk(obj[:,4],2)

    u = topTwo.values[0] - topTwo.values[1]

    return u


def cluster_lc(obj):


    return 1 - torch.max(obj[:,4])


def least_confidence(prediction, path, n=0 ):

    #result = [path.stem, 1.0,1.0,1.0,1.0]
    leastConfidence = 1

    for i, det in enumerate(prediction):
        for *xyxy, conf, cls in reversed(det):
            cn = int(cls) + 1           #classnumber
            if(cn>4):                   #wrong model
                print(cn)
                return 

            #result[cn] = min(float(conf), result[cn])
            leastConfidence = min(float(conf), leastConfidence)

    #return (result[0],result[1],result[2],result[3],result[4])
    return (path.stem, leastConfidence)




