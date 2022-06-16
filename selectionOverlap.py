from utils.acquisition import loadFile




#Calculating the selection overlap to determine how 
def selectionOverlap(selection1, selection2):

    s1 = loadFile(selection1)
    s2 = loadFile(selection2)
    if len(s1) != len(s2):
        print("Selection files not equal!")
        return 0

    overlap = 0
    size = len(s1)

    fileNames1 = [file[0] for file in s1]
    fileNames2 = [file[0] for file in s2]

    overlap = len(set(fileNames1).intersection(fileNames2))

    return overlap/size



if __name__ == "__main__":
    #sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/selectionLC.txt"
    #sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/selectionRND.txt"

    #3 Dropout Inferences
    #sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp454/acquisition/selection0.txt"
    #sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp455/acquisition/selection.txt"

    #14 Dropout Inferences
    sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/LU0/acquisition/selection.txt"
    sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/LU1/acquisition/selection.txt"

    print(selectionOverlap(sel1,sel2))
