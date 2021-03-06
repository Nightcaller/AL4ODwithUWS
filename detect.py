# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
from re import U
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync, apply_dropout

from utils.active_learning import random_sampling, uncertainty, least_confidence, location_stability, robustness, margin,entropy, location_uncertainty, cluster_entropy
from utils.al_helpers import save_text, plot_distribution, gaussian_noise, flip_predicitions, hungarian_clustering

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        dropout=10,
      #  al_leastConf=False,
       # al_random=False,
        al='none',
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)

    #for some reason locally following argument is a list and on collab its a string
    # --weights /path/to/model1 /path/to/model2
    if isinstance(weights,str):
        weights = weights.split()

    models = []
    if len(weights) == 1:
        weights = weights[0]
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
    else: 
        for weight in weights:
            model = DetectMultiBackend(weight, device=device, dnn=dnn)
            models.append(model)
    
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs



    col = 0
#########                                        #added 
    inferences = 1                 
    if al == "lu_d" or al == "entropy_d":
        inferences = dropout   
    if al == "lu_e" or al == "entropy_e":
        inferences = len(models)    
    if al == "ls":
        inferences = 7              # one reference image 6 levels of noise
    if al == "ral":
        inferences = 2              # reference + horizontal flip

#Overhead for any AL Strategies 
    if al != "none":                                
        save_acq = str(save_dir / 'acquisition' )
        (save_dir / 'acquisition').mkdir(parents=True, exist_ok=True)
        al_u = []


##########


    # Run inference#
    if al == "lu_e" or al == "entropy_e":
        for model in models:
            model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    else:
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

    dt, seen = [0.0, 0.0, 0.0], 0

    ##### added progress bar over dataset
    pbar = enumerate(dataset)
    pbar = tqdm(pbar, total=len(dataset), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
    ######

    for i, (path, im, im0s, vid_cap, s) in pbar:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        
        predictions = []                        #added
        confidences = []                        #added
        if al == "ls" or al == "ral":
            im_copy = im
        # Inference 
        #added dropout loop for iference with turned on dropout layers
        for i in range(0,inferences):                       #added
            if al == "random":
                break
            if al == "lu_e" or al == "entropy_e":
                model = models[i]
            if al == "ls":
                im = gaussian_noise(im_copy, 8 * i)
            if al == "ral" and i > 0:
                im = im.flip(-1)
            if (al == "lu_d" or al == "entropy_d") and i >= 1:
                model.apply(apply_dropout) 

            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS added returning class confidences for each class
            pred, confs = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, det=True)
            dt[2] += time_sync() - t3

            if al == "ral" and i > 0:
                pred = flip_predicitions(im, pred)

            #saving all inference runs
            predictions.append(pred[0])           #added
            confidences.append(confs[0])           #added
            
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


        # Select random    
        if al == "random":
            al_u.append((Path(path).stem, random_sampling()))
        if al == "lu_d":
            result = location_uncertainty(predictions, confidences)
            if result == None:
                imageScore = 0
                objectScores = [0]
            else:
                #print(result)
                imageScore, objectScores = result

            al_u.append((Path(path).stem, imageScore))
            for i, pred in enumerate(predictions[0]):
                pred[4] = objectScores[i]

        if al == "lu_e":
            
            result = location_uncertainty(predictions, confidences, ensemble=True)
            if result == None:
                imageScore = 0
                objectScores = [0]
            else:
                #print(result)
                imageScore, objectScores = result
                al_u.append((Path(path).stem, imageScore))
                for i, pred in enumerate(predictions[0]):
                    pred[4] = objectScores[i]

                
        if al == "lc":
            al_u.append((Path(path).stem, least_confidence(pred[0])))
        if al == "margin":
            al_u.append((Path(path).stem, margin(confs[0])))
        if al == "entropy": 
            al_u.append((Path(path).stem, entropy(confs[0])))
        if al == "entropy_d":
            al_u.append((Path(path).stem, cluster_entropy(predictions, confidences)))
        if al == "entropy_e":
            al_u.append((Path(path).stem, cluster_entropy(predictions, confidences)))
        if al == "ls":
            al_u.append((Path(path).stem, location_stability(predictions)))
        if al == "ral":
            al_u.append((Path(path).stem, robustness(predictions, confidences)))



        #added
        im0 = im0s.copy()   # to annotate every pred on same image  
        objects, confPairs = hungarian_clustering(predictions, confidences, 0.3)
        # Process predictions

        for objN ,prediction in enumerate(objects[::-1]):

            for i, det in enumerate(prediction):  # per image

                det = det[None,:]

                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    #p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  
                    p, im0, frame = path, im0, getattr(dataset, 'frame', 0)         #added - draw on the same image

                p = Path(p)  # to Path

                #if al == "dropout" and singleObject:
                #    save_path = str(save_dir) + "/" + p.stem + "_" + str(objN) + ".jpg"  # one image per Object
                #else:
                #    save_path = str(save_dir / p.name)  # im.jpg

                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
    
    #removed 
   #             s += '%gx%g ' % im.shape[2:]  # print string           #removed Logger

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results                 #removed Logger
                    #for c in det[:, -1].unique():
                    #    n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        #i == (len(prediction)-1) => so only the mean label is saved 
                        if save_txt and i == (len(prediction)-1):  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            
                            c = int(cls)  # integer class

                            # mean label is annotated
                            #if(i == (len(prediction)-1) and  al == "dropout"):
                            #    label = f'{len(prediction)-1} | {conf:.2f}' 
                            #    annotator = Annotator(im0, line_width=2, example=str(names))
                            #    annotator.box_label(xyxy, label, (0,0,0)) #black box
                            colorsAr= [
                                (66, 135, 245),(252, 30, 3),(3, 252, 132),
                                (33, 16, 115),(3, 252, 132),
                                (252, 0, 122),(0,0,0),(252, 0, 206),
                                (3, 244, 252),(255,255,255),(235, 64, 52),
                                (252, 111, 3),(140, 13, 127),
                                (50,0,50),(3, 194, 252),
                                (3, 148, 252),(3, 74, 252),
                                (66, 135, 245),(252, 30, 3),(3, 252, 132),
                                (3, 244, 252),(255,255,255),
                                (252, 0, 122),(0,0,0),(252, 0, 206),
                                (50,0,50),(3, 194, 252),
                                (3, 244, 252),(255,255,255),(235, 64, 52),
                                (66, 135, 245),(252, 30, 3),(3, 252, 132),
                                (33, 16, 115),(3, 252, 132),
                                (252, 0, 122),(0,0,0),(252, 0, 206),
                                (3, 244, 252),(255,255,255),(235, 64, 52),
                                (252, 111, 3),(140, 13, 127),
                                (50,0,50),(3, 194, 252),
                                (3, 148, 252),(3, 74, 252),
                                (66, 135, 245),(252, 30, 3),(3, 252, 132),
                                (3, 244, 252),(255,255,255),
                                (252, 0, 122),(0,0,0),(252, 0, 206),
                                (50,0,50),(3, 194, 252),
                                (3, 244, 252),(255,255,255),(235, 64, 52),
                                (66, 135, 245),(252, 30, 3),(3, 252, 132),

                            ]
                            
                            #Reference label
                            if(objN == len(predictions)-1 and  (al == "lu_d" or al == "entropy_d" or al == "ral" )):
                                #label = f'U(O): {conf:.4f}' 
                                #annotator = Annotator(im0, line_width=2, example=str(names))
                                #annotator.box_label(xyxy, label, (50,205,50)) #limegreen box
                                
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colorsAr[objN])
                                
                            #Dropoutlabels
                            else:
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                #nnotator.box_label(xyxy, label, color=colors(c, True))
                                annotator.box_label(xyxy, label, color=colorsAr[objN])
                                
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
            # Print time (inference-only)
        
        
            ##
            #EXPORT IMAGE PRINTING


            # Stream results
            
        if view_img:
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            im0 = annotator.result()
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

                #####


        #moved logger so it only logs after it's finished with one Image
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')          #removed Logger

    # Print results
    #t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
    #added
     #sort & plot & save txt
    if al != "none":    
        al_u.sort(key=lambda x:x[1])
        save_text(al_u, save_acq, "uncertainty")

        #Random                                           
        if al == "random":
            plot_distribution(al_u, save_acq, "Random_Sampling", names)
        #Location Uncertainty 
        if al == "lu_d":
            plot_distribution(al_u, save_acq, "Location_Uncertainty_Dropout", names)
        if al == "lu_e":
            plot_distribution(al_u, save_acq, "Location_Uncertainty_Ensembles", names)
        #least Confidence
        if al == "lc":
            plot_distribution(al_u, save_acq, "Least_Confidence", names)
        #location stability
        if al == "ls":
            plot_distribution(al_u, save_acq, "Location_Stability", names)
        if al == "ral":
            plot_distribution(al_u, save_acq, "Robustness", names)
        if al == "margin":
            plot_distribution(al_u, save_acq, "Margin", names)
        if al == "entropy":
            plot_distribution(al_u, save_acq, "Entropy", names)
        if al == "entropy_d":
            plot_distribution(al_u, save_acq, "Entropy_Dropout", names)
        if al == "entropy_e":
            plot_distribution(al_u, save_acq, "Entropy_Ensembles", names)
    ##########

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    pwd = "/Users/mhpinnovation/Documents/Daniel/Master/detector/bookish-carnival/models/"
    #w = pwd + "11.pt"
    w = [pwd + "73.pt",pwd + "42.pt",pwd + "11.pt" ,pwd + "42.pt"]
        #
    parser.add_argument('--weights', nargs='+', type=str, default=w, help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS t threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

##########
    #added arguments for AL Strategies
    parser.add_argument('--dropout', type=int, default=10, help='how many inferences should be run in case of dropout al detection') #added
    # parser.add_argument('--al_random', action='store_true', help='activate random acquisition values') #added
    # parser.add_argument('--al_leastConf', action='store_true', help='activate least confidence acquisition values') #added
    parser.add_argument('--al', default='lu_e', help='activate least confidence acquisition values') #added
##########

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

