from utils.acquisition import loadFile




#Calculating the selection overlap to determine how 
def selectionOverlap(drops):

    drops = str(drops)

    sel0 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_0LU_d/uncertainty.txt"
    sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_1LU_d/uncertainty.txt"
    sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_2LU_d/uncertainty.txt"
    sel3 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_3LU_d/uncertainty.txt"
    sel4 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_4LU_d/uncertainty.txt"


    s1 = loadFile(sel0)
    s2 = loadFile(sel1)
    s3 = loadFile(sel2)
    s4 = loadFile(sel3)
    s5 = loadFile(sel4)
    if len(s1) != len(s2) or len(s1) != len(s3):
        print("Selection files not equal!")
        return 0

    
    size = len(s1)

    fileNames1 = [file[0] for file in s1]
    fileNames2 = [file[0] for file in s2]
    fileNames3 = [file[0] for file in s3]
    fileNames4 = [file[0] for file in s4]
    fileNames5 = [file[0] for file in s5]

    split = 800
    fileNames1 = fileNames1[split:]
    fileNames2 = fileNames2[split:]
    fileNames3 = fileNames3[split:]
    fileNames4 = fileNames4[split:]
    fileNames5 = fileNames5[split:]

    overlap = size - split
    overlapList = []

    overlapList.append(len(set(fileNames1).intersection(fileNames2)) / overlap)
    overlapList.append(len(set(fileNames1).intersection(fileNames3)) / overlap)
    overlapList.append(len(set(fileNames1).intersection(fileNames4)) / overlap)
    overlapList.append(len(set(fileNames1).intersection(fileNames5)) / overlap)

    overlapList.append(len(set(fileNames2).intersection(fileNames3)) / overlap)
    overlapList.append(len(set(fileNames2).intersection(fileNames4)) / overlap)
    overlapList.append(len(set(fileNames2).intersection(fileNames5)) / overlap)

    overlapList.append(len(set(fileNames3).intersection(fileNames4)) / overlap)
    overlapList.append(len(set(fileNames3).intersection(fileNames5)) / overlap)

    overlapList.append(len(set(fileNames4).intersection(fileNames5)) / overlap)

    


    avgOverlap = sum(overlapList) / len(overlapList)
    print(overlapList)
    print(f'Average: {avgOverlap} | Min: {avgOverlap-min(overlapList) }| Max: {max(overlapList) - avgOverlap}')
    
    #print(f'12: {o12} | 13: {o13} | 23: {o23}')



def selectionOverlapDuo(selection1, selection2):

    s1 = loadFile(selection1)
    s2 = loadFile(selection2)

    #if len(s1) != len(s2):
    #    print("Selection files not equal!")
    #    return 0

    overlap = 0
    size = len(s1)

    fileNames1 = [file[0] for file in s1]
    fileNames2 = [file[0] for file in s2]


    overlap12 = len(set(fileNames1).intersection(fileNames2))

    o12 = overlap12/size


    return o12



if __name__ == "__main__":
    #sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/selectionLC.txt"
    #sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/selectionRND.txt"

    #3 Dropout Inferences
    #sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp454/acquisition/selection0.txt"
    #sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp455/acquisition/selection.txt"

    #14 Dropout Inferences
    

    #Full selection list 
    #sel90Ent = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyEntropy/selection.txt"
    #sel90Ral = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyRAL/selection.txt"
    #sel90Margin = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyMargin/selection.txt"
    #sel90LUD = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyLUD/selection.txt"
    

    sel90Ent = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyEntropy/0ninetyEntropy0/acquisition/selection.txt"
    sel90Ral = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyRAL/0ninetyRAL0/acquisition/selection.txt"
    sel90Margin = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyMargin/0ninetyMargin0/acquisition/selection.txt"
    sel90LUD = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyLUD/0ninetyLUD0/acquisition/selection.txt"
    sel90LC = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyLC/0ninetyLC0/acquisition/selection.txt"
    sel90LS = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyLS/0ninetyLS0/acquisition/selection.txt"
    sel90LUE = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0ninetyLU_Eall/detect/0ninetyLU_E0/acquisition/selection.txt"
    sel90EntD = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0nintyEntDrop/0nintyEntDrop0/acquisition/selection.txt"
    sel90EntE = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0nintyEntropy_E0/acquisition/selection.txt"

    
    sel50Ent = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0fiftyEntropy/0fiftyEntropy0/acquisition/selection.txt"
    sel50Ral = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0fiftyRAL/0fiftyRAL0/acquisition/selection.txt"
    sel50Margin = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0fiftyMargin/0fiftyMargin0/acquisition/selection.txt"
    sel50LUD = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0fiftyLUD/0fiftyLUD0/acquisition/selection.txt"
    sel50LC = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0fiftyLC/0fiftyLC0/acquisition/selection.txt"
    sel50LS = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/0fiftyLS/0fiftyLS0/acquisition/selection.txt"
    sel50LUE = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/fiftyLU_Eall/detect/0fiftyLU_E0/acquisition/selection.txt"
    sel50Ent_E = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/train/fiftyEntropy_E/selection.txt"
    

    

    #print("Ent / Margin " + str(selectionOverlapDuo(sel90Ent,sel90EntE)))
    #print("Ent / Ral " + str(selectionOverlapDuo(sel90Ent,sel90Ral)))
    #print("Ral / Margin " + str(selectionOverlapDuo(sel90Ral,sel90Margin)))

    #print("LUD / Margin " + str(selectionOverlapDuo(sel90LUD,sel90Margin)))
    #print("LUD / Ral " + str(selectionOverlapDuo(sel90LUD,sel90Ral)))
    #print("LUD / Ent " + str(selectionOverlapDuo(sel90Ent,sel90LUD)))


    #print("LUD / LC " + str(selectionOverlapDuo(sel90LUD,sel90LC)))

    print("LUE / ENT_E " + str(selectionOverlapDuo(sel90LUE,sel90EntE)))
    print("LC / ENT_E " + str(selectionOverlapDuo(sel90LC,sel90EntE)))
    print("margin / ENT_E " + str(selectionOverlapDuo(sel90Margin,sel90EntE)))
    print("Ent / ENT_E " + str(selectionOverlapDuo(sel90Ent,sel90EntE)))
    
    print("LS / ENT_E " + str(selectionOverlapDuo(sel90LS,sel90EntE)))
    print("RAL / ENT_E " + str(selectionOverlapDuo(sel90Ral,sel90EntE)))
    print("LUD / ENT_E " + str(selectionOverlapDuo(sel90LUD,sel90EntE)))

    print("ent_D / ENT_E " + str(selectionOverlapDuo(sel90EntD,sel90EntE)))
    







    #for drops in range(25,26):     
   ##     print(f'Drops: {drops}')
    #    selectionOverlap(drops)
    #    print("########################")
