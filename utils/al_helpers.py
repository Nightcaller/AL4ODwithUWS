import matplotlib.pyplot as plt
import numpy as np


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
        print("###########")
        print(len(values))
        print("###########")
        #subplots
        fig, axs = plt.subplots(4)
        fig.suptitle(type, fontsize=18)
        for i in range(1,len(values[0])):
            values.sort(key=lambda x:x[i])          
            value = np.array([x[i] for x in values])
            value[ value==1.0] = 0.0             #don't plot default
            
            axs[i-1].plot(value)
            axs[i-1].set_title('Class' + classnames[i])   #parse class name
            
        
        plt.ylim[0,1]
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



#Edit for more Lines 
def save_text(values, save_dir, type):
    save_dir = save_dir + "/" + type + ".txt"
    with open(save_dir, 'a') as f:
        for value in values:
            f.write( value[0] + " " + str(value[1]) + "\n")
