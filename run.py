import torch
import os
import time
import random
import numpy as np

def set_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def runTraining(args):
    import train
    

    opt = train.parse_opt(True)
    for k,v in args.items():
        setattr(opt, k, v)

    train.main(opt)


def runDetect(args):
        import detect 
        

        opt = detect.parse_opt(True)
        
        for k,v in args.items():
            setattr(opt, k, v)
        print(opt)

        detect.main(opt)


def runAcquisition(name):
  import utils.acquisition as acq
  source = "/content/datasets/waymo/unlabeled"
  target = "/content/datasets/waymo/training"
  threshold = 0.05
  acqFile = "/content/bookish-carnival/runs/detect/" + name + "/acquisition"
  modes = ["ssl"]

  
  acq.selection(acqFile, source, target, threshold, modes, n=2500)


def runOracle(cycle, experimentName):
    import utils.oracle as oracle
    
    gtPath = "/content/datasets/waymo/training"
    acqPath = "/content/bookish-carnival/runs/detect/" + cycle + "/acquisition"
    savePath = "/content/bookish-carnival/runs/train/" + experimentName
    predPath = "/content/bookish-carnival/runs/detect/" + cycle
    
    return oracle.autOracle(gtPath, savePath, acqPath,predPath, cycle)






def runActiveLearning(al, experimentName):

    epochs = 250

    trainArgs = {
    "rect" : True,
    "batch": "32",
    "epochs": epochs,
    "data": "/content/bookish-carnival/data/waymo.yaml",
    "weights": "/content/bookish-carnival/models/11.pt",
    "cfg": "/content/bookish-carnival/models/DropBeforeDetect.yaml",
    "name": experimentName,
    "exist_ok": True,
    "fixedStop": True,
    "cache": "ram",
    }

    detectArgs = {
    "al" : al,
    "weights": "/content/bookish-carnival/models/11.pt",
    "source": "/content/datasets/waymo/unlabeled/images",
    "save_txt": True,
    "exist_ok": True,
    "nosave": True,

    }

    labelingTime = 0
    phase = 0

    while phase < 5:

        cycle = experimentName + str(phase)

        detectArgs["name"] = cycle
        runDetect(detectArgs)
        runAcquisition(cycle)
        labelingTime += runOracle(cycle, experimentName)




        runTraining(trainArgs)
        trainArgs["weights"] = "/content/bookish-carnival/runs/train/" + experimentName + "/weights/best.pt"
        phase += 1
        trainArgs["resume"] = True



def runActiveLearningEnsemble(al, name):


    seeds = [3,7,11,42,73]
    ensembleWeights = ""
    
    epochs = 250
    labelingTime = 0

    for seed in seeds:
        ensembleWeights +=  " /content/bookish-carnival/models/" + str(seed) + ".pt"

    trainArgs = {
        "rect" : True,
        "batch": "32",
        "epochs": epochs,
        "data": "/content/bookish-carnival/data/waymo.yaml",
        "weights": "",
        "cfg": "/content/bookish-carnival/models/DropBeforeDetect.yaml",
        "name": name,
        "exist_ok": True,
        "fixedStop": True,
        "cache": True,
        }

    detectArgs = {
        "al" : al,
        "weights": ensembleWeights,
        "source": "/content/datasets/waymo/unlabeled/images",
        "save_txt": True,
        "exist_ok": True,
        "nosave": True,
        }

    phase = 0

    while phase < -1:

        cycle = name + str(phase)
        detectArgs["name"] = cycle


        runDetect(detectArgs)

        runAcquisition(cycle)
        labelingTime += runOracle(cycle, cycle)

        for seed in seeds:
            experimentName = str(seed) + name
            trainArgs["name"] = experimentName
            
            set_seed(seed)

            if phase == 0:
                trainArgs["weights"] = "/content/bookish-carnival/models/" + str(seed) + ".pt"
            else:
                trainArgs["weights"] =  "/content/bookish-carnival/runs/train/" + experimentName + "/weights/best.pt "
                trainArgs["resume"] = "/content/bookish-carnival/runs/train/" + experimentName + "/weights/best.pt"


            runTraining(trainArgs)
        
        phase += 1
        ensembleWeights = ""
        for seed in seeds:      
            experimentName = str(seed) + name 
            ensembleWeights +=  "/content/bookish-carnival/runs/train/" + experimentName + "/weights/best.pt "
            detectArgs["weights"] = ensembleWeights

      





if __name__ == "__main__":
    al = "entropy"
    experimentName = "0newENTROPY"
    runActiveLearning(al, experimentName)
