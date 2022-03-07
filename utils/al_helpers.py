import matplotlib.pyplot as plt
import numpy as np
import cv2 
from utils.plots import Annotator
import torch


#depricated
'''
def sort_uncertainty_values(fileName, path, type, plot=True):

    file1 = open('myfile.txt', 'r')
    Lines = file1.readlines()
    
    uncertainty_values = []
    image_names = []

    for line in Lines:
        line = line.strip()
        splitted = line.split()
        image_names.append(splitted[0])
        uncertainty_values.append(float(splitted[1]))


    #sort by uncertainty_valuees
    uncertainty_values, image_names = zip(*sorted(zip(uncertainty_values, image_names), key=lambda x: x[0]))

    uncertainty_values = np.array(uncertainty_values)

    plot_distribution(uncertainty_values, path)

'''


def plot_distribution(values, path, type, classnames):

    #class names

    if(len(values[0]) > 2):
       
        #subplots
        fig, axs = plt.subplots(len(values[0]) - 1)
        fig.suptitle(type, fontsize=18)
    
        #print("###########")
        #print(len(values[0]))

        #print("###########")
        for i in range(1,len(values[0])):
            values.sort(key=lambda x:x[i])          
            value = np.array([x[i] for x in values])
            value[ value==1.0] = 0.0             #don't plot default
            
            axs[i-1].plot(value)

            axs[i-1].set_title('Class: ' + classnames[i-1])   #parse class name
            
        
        #plt.ylim([0,1])
        plt.savefig(path + "/" +  type + ".jpg")
               
    else:
        values = np.array([x[1] for x in values])
        fig = plt.figure()
        plt.plot(values)
        fig.suptitle(type, fontsize=18)
        plt.xlabel("# of Image", fontsize=14)
        plt.ylabel("Uncertainty Value", fontsize=14)
        
        plt.savefig(path + "/" +  type + ".jpg")
        plt.close()





# Check expPath for 0,1,2,3,4 experiments with file=results.txt or oracle.txt and plot)
def plot_results(expPaths, subPaths ,file, savePath, n, colors, xLabel, labels):

    import os

    fig, ax = plt.subplots()
    plots = []

    
    

    for j, exp in enumerate(expPaths):
 
        results = []
        lasts = []
        lastYs = []
        for dir in subPaths[j]:
            path = exp + "/" + dir + "/" + file   
            
            if os.path.exists(path):
                with open(path) as f:
                    names = f.readline().split(",")
                    results.append([x.split(",") for x in f.read().strip().splitlines()])
                    

        if(not results):
            print(path)
            print("nothing here")
        
            return 

        
        name = names[n].split("/")
        if(len(name) > 1):
            name = name[1]
        else:
            name = name[0]

        for k, x in enumerate(results): 
            lasts.append(x[-1])
            lastYs.append( (len(x) * (k+1)) -1 )
            


        #for i, result in enumerate(results):
        result = sum(results, [])
        values = np.array([float(x[n]) for x in result])
        #epochs = np.array([float(x[0]) for x in result]) + len(values) * i +1  #weg machen
        #lastY =  len(values) * (i + 1)
   
        #last = values[-1]
        test = np.array([[   0.035823  ,  0.070962 ,   0.094206   ,  0.10682    , 0.11933],[     0.080872  ,    0.1108   ,  0.12543  ,   0.13263  ,   0.13886], [0.16512] ])
        plots.append(plt.plot(test[j], values, color=colors[j], label=labels[j]))
        #plots.append(plt.plot( values, color=colors[j], label=labels[j]))
        #[    0.32781]
        
        lastValues = np.array([float(x[n]) for x in lasts])
        print(lastValues)
        #plt.plot(lastYs, lastValues, marker="x", color="black", linestyle="None")
        plt.plot(test[j], lastValues, marker="x", color="black", linestyle="None")

    ax.legend()
    fig.suptitle(name, fontsize=18)
    plt.xlabel(xLabel, fontsize=14)
    plt.ylabel(name, fontsize=14)

    
    #plt.xlim([0.1,0.35])
    
    plt.savefig(savePath + "/" +  name + ".jpg")
    plt.close()









#Edit for more Lines 
def save_text(values, save_dir, fileName):
    save_dir = save_dir + "/" + fileName + ".txt"
    with open(save_dir, 'a') as f:
        for value in values:
            
            f.write( value[0] + " " + str(value[1]) + "\n")




def annotate_image(path, savePath, gtBoxes, predBoxes, ious=None):

    image = cv2.imread(path)
    annotator = Annotator(image, line_width=1)
    gn = torch.tensor(image.shape)[[1, 0, 1, 0]]


    for i, box in enumerate(gtBoxes):
        if isinstance(ious, torch.Tensor): 
            hit = float(ious[i])
            if(hit>0.90): 
                color = (57,255,20)
            elif(hit>0.80):
                color = (0,255,255)
            else: 
                color = (0,0,190)

            label = str(int(box[4])) + " | " + str(float(ious[i]))
            
            annotator.box_label(box[:4]*gn,label , color)
        else:
            label = str(int(box[4]))
            annotator.box_label(box[:4]*gn,label , (57,255,20))


    for box in predBoxes:
        label = str(int(box[4]))
        annotator.box_label(box[:4]*gn,label , (186,0,0))

    return cv2.imwrite(savePath, image)

