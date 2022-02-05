import random
import torch

from utils.metrics import box_iou

def random_sampling(): 

    return random.uniform(0, 1)

def uncertainty(predictions, path, imgSize , threshold_iou=0.8):
    
    objects = []
    first = True
     
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
            

            
            for i, d in enumerate(det):
                ious = []
                for object in objects:
                    ious.append(torch.max(box_iou(d[None,:4], object[:,:4])))


                index = ious.index(max(ious))

                if(sum(ious) == 0):   
                    objects.append(d[None,:])
                    continue
                if(d[5] == objects[index][0][5]):
                    objects[index] = torch.cat((objects[index], d[None,:]), 0) 
                else:
                    objects.append(d[None,:])

    for i, obj in enumerate(objects):
        
        mean = torch.mean(obj, 0)
        objects[i] = torch.cat((obj, mean[None,:]), 0) 
        #print(objects)
    return objects
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
def cluster(object):
    
    
    
    
    
    return
 


def least_confidence(prediction, path, n=0 ):

    result = [path.stem, 1.0,1.0,1.0,1.0]

    for i, det in enumerate(prediction):
        for *xyxy, conf, cls in reversed(det):
            cn = int(cls) + 1           #classnumber
            if(cn>4):                   #wrong model
                print(cn)
                return 

            result[cn] = min(float(conf), result[cn])
            

    return (result[0],result[1],result[2],result[3],result[4])





