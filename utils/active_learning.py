from platform import python_version_tuple
import random
from sklearn.cluster import cluster_optics_dbscan
import torch
import numpy as np

from utils.metrics import box_iou
from utils.al_helpers import js_divergence, kl_divergence, hungarian_clustering


############################################################################
############################################################################
#########  CLASSIC METHODS #########
############################################################################
############################################################################

def random_sampling(): 

    return random.uniform(0, 1)

#class agnostic least confidence
def least_confidence(prediction):

    if len(prediction) <= 0:
        return 0
        
    return float(1 - torch.min(prediction[:,4]))


def margin(confs):
    
    if(confs.size()[0] <=1 ):
        return 0

    topTwo = torch.topk(confs,2)

    return 1 - (sum(topTwo.values[:,0]) - sum(topTwo.values[:,1]))/len(topTwo.values)


def entropy(confs):

    cuda = torch.cuda.is_available()
    try:
        confs[0]
    except:
        return 0

    if cuda:
        size = torch.tensor(len(confs)).to('cuda:0')
        classes = torch.tensor(len(confs[0])).to('cuda:0')
    else:
        size = torch.tensor(len(confs[0]))
        classes = torch.tensor(len(confs[0]))
    
    entropies = 0

    for conf in confs:
        logProbs = torch.mul(conf ,torch.log2(conf))
        numerator = torch.sub(torch.tensor(0), torch.sum(logProbs))
        denom = torch.log2(classes)

        entropies += torch.div(numerator, denom)
    
  

    return entropies/size


def cluster_entropy(predictions, confidences):

    _, confPairs = hungarian_clustering(predictions, confidences)

    if len(confPairs) < 1:
        return 0

    entropies = 0
    for confs in confPairs:
        entropies += entropy(confs)


    return entropies / len(confPairs)



############################################################################
############################################################################
###########  ADVANCED METHODS  #############################################
############################################################################
############################################################################



def location_uncertainty(predictions, confidences):

    objects, confPairs = hungarian_clustering(predictions, confidences, 0.3)

    inferences = len(predictions)
    if len(objects) < 1:
        return 0
    
    sumLU = 0
    maxLU = 0
    avgLU = 0

    

    for i, preds in enumerate(objects):
        if len(preds) < inferences/2:
            continue

        jsDiv = sum(js_divergence(confPairs[i][0], confPairs[i][j]) for j in range(1,len(preds)-1 )) / len(preds)
        #meanBox = torch.mean(preds, 0)
        lu = (1 - (torch.sum(box_iou(preds[None, 0,:4], preds[1:,:4])) / len(preds)))
        #lu = (1 - (torch.sum(box_iou(meanBox[None,:4], preds[:,:4])) / len(preds)))
        ent = entropy(confPairs[i]) 

        lu = lu * jsDiv
        #lu = lu * ent
        sumLU += lu
        maxLU = max(lu,maxLU)
        
        
    avgLU = sumLU / len(objects)

    weightedLU = (avgLU + maxLU + sumLU) / 3


    return (avgLU + maxLU) / 2
    #return maxLU
    #return avgLU

def location_stability(predictions):

    objects = hungarian_clustering(predictions)

    if len(objects) < 1:
        return 0

    sumB0P, sumP, sb, maxU = 0, 0, 0, 0

    for obj in objects:
        if len(obj) > 1:
            sb = torch.sum(box_iou(obj[None,0,:4],obj[1:,:4] )) / 6
            pmax = obj[0][4]                # obj[0] => Reference Box ; [4] => Pmax
            sumB0P += pmax * sb                      
            sumP += pmax
            u = least_confidence(obj) 

            if u > maxU:
                maxU = u

    if sumP == 0:
        return 0


    return (maxU - (sumB0P/sumP))


def robustness(predictions, confidences):

    objPairs, confPairs = hungarian_clustering(predictions, confidences)

    if len(objPairs) == 0:
        return 0

    classConsistency, entrop, validPairs = 0, 0, 0


    for i, pair in enumerate(confPairs):

        if len(pair) <= 1:
            continue
        if len(pair) >= 3:          #only use maxIoU pairs 
            maxIoUIndex = torch.argmax(box_iou(objPairs[i][None, 0,:4], objPairs[i][1:,:4] ))
            pair = [pair[0], pair[maxIoUIndex+1]]

        pq = kl_divergence(pair[0],pair[1])
        qp = kl_divergence(pair[1],pair[0])

        
        classConsistency += 0.5 * (pq+qp)
        entrop += entropy(pair)
        validPairs += 1

    if validPairs == 0:
        return 0

    classConsistency = classConsistency  / validPairs
    entrop = entrop  / validPairs

    return classConsistency * entrop











    
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################

############################################################
############################################################
############################################################
############################################################
############################################################



#Deprecated
def location_stability_old(predictions):


    objects = []
    first = True
    ls = []
    count = []
    maxU = 0

    #cluster all predicitions into objects 
    for det in predictions:
        #for det in prediction:

            #for *xyxy, conf, cls in reversed(det):                       # det[:,:4] => BB ; det[:,4] => Confidence, det[:,5] => Class    
            
            # initial reference boxes without noise (B0)
        if(first): 
            if(len(det) == 0):
                return 0
            for box in det:
                objects.append(box[None,:])
                ls.append(0)
                count.append(0)
                
            first = False
            continue


        if(len(det) == 0): 
            continue

        u = 1 - torch.min(det[:,4])
        if u > maxU:
            maxU = u

        # enumerate corresponding boxes with noise C(B0) and sum the iou(B0,C(B0)) 
        for i, box in enumerate(det):
            ious = []
            for object in objects:
                ious.append(torch.max(box_iou(box[None,:4], object[:,:4])))

            
            
            #assign BB to max overlap 
            maxIoU = max(ious)
            index = ious.index(maxIoU) #index of corresponding box

            ls[index] += maxIoU
            count[index] += 1

    
    for i, _ in enumerate(ls):
        if count[i] > 0:
            ls[i] = ls[i] / count[i]

    sumB0P = sum((x[0][4]*ls[i]) for i, x in enumerate(objects))
    sumP = sum(x[0][4] for x in objects)

    return 0.5 + (maxU - (sumB0P/sumP))

#Deprecated
def robustness_old(predictions, confs):

    objects = []
    first = True
    pairs = []

    #cluster all predicitions into objects 
    for det in predictions:
        #for det in prediction:

            if(len(det) == 0):
                continue
            #for *xyxy, conf, cls in reversed(det):                       # det[:,:4] => BB ; det[:,4] => Confidence, det[:,5] => Class    
            
            if(first):
                for box in det:
                    objects.append(box[None,:])
                first = False
                pairs = [-1] * len(objects)
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
                    continue
                if(d[5] == objects[index][0][5] ):   #add to existing cluster
                    objects[index] = torch.cat((objects[index], d[None,:]), 0) 
                    pairs[index] = i

    classCon = 0
    ent = 0

    validPairs = 0

    for i, pair in enumerate(pairs):
        if pair == -1:
            continue

        pq = kl_divergence(confs[0][i],confs[1][pair])
        qp = kl_divergence(confs[1][pair],confs[0][i])

        classCon += 0.5 * (pq+qp)
        ent += entropy([confs[0][i],confs[1][pair]])
        validPairs += 1

    if validPairs == 0:
        return 0

    classCon = classCon  / validPairs
    ent = ent  / validPairs

    return ent * classCon

#Deprecated
# BB Clustering by Hungarian Method
def uncertainty(predictions, mode="DBScan" , threshold_iou=0.3):
    
    objects = []
    uAll = []
    first = True

    #cluster all predicitions into objects 
    for prediction in predictions:
        for det in prediction:

            if(len(det) == 0):
                continue
            #for *xyxy, conf, cls in reversed(det):                       # det[:,:4] => BB ; det[:,4] => Confidence, det[:,5] => Class    
            
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

def cluster_entropy_old(obj):
      

    if(obj.size()[0] <=1 ):
        return 0


    ''' 
    probs = obj[:,4].cpu().numpy()

    logProbsCPU = probs * np.log2(probs)
    numeratorCPU = 0 - sum(logProbsCPU)
    denomCPU = np.log2(probs.size)

    entropyCPU = numeratorCPU / denomCPU
    '''

    cuda = torch.cuda.is_available()
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


    return 1 - torch.min(obj[:,4])




def old_location_uncertainty(predictions, confidences):

    objects, confPairs = hungarian_clustering(predictions, confidences)

    inferences = len(predictions)
    if len(objects) < 1:
        return 0
    
    sumLU = 0
    maxLU = 0
    avgLU = 0

    for i, preds in enumerate(objects):
        if len(preds) < inferences/2:
            continue


        meanBox = torch.mean(preds, 0)
        lu = (1 - (torch.sum(box_iou(meanBox[None,:4], preds[:,:4])) / inferences))
        #lu = (1 - (torch.sum(box_iou(meanBox[None,:4], preds[:,:4])) / len(preds)))
        ent = entropy(confPairs[i]) 

        lu = (lu * ent) 
        sumLU += lu
        if maxLU < lu:
            maxLU = lu
        
        
    avgLU = sumLU / len(objects)

    weightedLU = (avgLU + maxLU + sumLU) / 3


    return (avgLU + maxLU) / 2

