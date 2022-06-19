from utils.acquisition import loadFile




#Calculating the selection overlap to determine how 
def selectionOverlap(selection1, selection2, selection3):

    s1 = loadFile(selection1)
    s2 = loadFile(selection2)
    s3 = loadFile(selection3)
    if len(s1) != len(s2) or len(s1) != len(s3):
        print("Selection files not equal!")
        return 0

    overlap = 0
    size = len(s1)

    fileNames1 = [file[0] for file in s1]
    fileNames2 = [file[0] for file in s2]
    fileNames3 = [file[0] for file in s3]

    overlap12 = len(set(fileNames1).intersection(fileNames2))
    overlap13 = len(set(fileNames1).intersection(fileNames3))
    overlap23 = len(set(fileNames2).intersection(fileNames3))

    o12 = overlap12/size
    o13 = overlap13/size
    o23 = overlap23/size

    avgOverlap = (o12+o13+o23)/3

    print(avgOverlap)
    print(f'12: {o12} | 13: {o13} | 23: {o23}')






if __name__ == "__main__":
    #sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/selectionLC.txt"
    #sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/selectionRND.txt"

    #3 Dropout Inferences
    #sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp454/acquisition/selection0.txt"
    #sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/runs/detect/exp455/acquisition/selection.txt"

    #14 Dropout Inferences
    drops = "10"
    sel0 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_0LU_d/selection.txt"
    sel1 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_1LU_d/selection.txt"
    sel2 = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/overlap/DROPOUT"+ drops + "_2LU_d/selection.txt"
    

    selectionOverlap(sel0,sel1,sel2)
