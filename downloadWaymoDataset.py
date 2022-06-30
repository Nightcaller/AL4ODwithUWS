'''
rm -rf waymo-od > /dev/null
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od && git branch -a
cd waymo-od && git checkout remotes/origin/master
pip3 install --upgrade pip

pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
'''


import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import cv2
from google.colab import auth
import csv

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def getImageAndLabel(camera_image, camera_labels, layout, f, cmap=None,):

    imageTensor = tf.image.decode_jpeg(camera_image.image)
    
    imageHeight = imageTensor.shape[0]
    imageWidth = imageTensor.shape[1]

    # Draw the camera labels.
    for idx,camera_labels in enumerate(frame.camera_labels):
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
        
            labelNumber = label.type - 1
            
            f.write(str(labelNumber) + " ")
            f.write("%.5f" % (label.box.center_x / imageWidth) + " ")
            f.write("%.5f" % (label.box.center_y / imageHeight) + " ")
            f.write("%.5f" % (label.box.length / imageWidth) + " ")
            f.write("%.5f" % (label.box.width / imageHeight) + "\n")
        


def parseFile(filename, filenumber):
  
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    imageCount = 0
    frame = open_dataset.Frame()

    for indexData, data in enumerate(dataset):
        
        frame.ParseFromString(bytearray(data.numpy()))    
        
        if(indexData % 30 == 0):
            for indexImage, image in enumerate(frame.images):
                imageCount += 1
                name = ("%04d" % filenumber) + ("%04d" % indexData) + ("%04d" % indexImage) 
                
                f = open("/content/drive/MyDrive/Waymo/training/labels/" +  name + ".txt", "w")
                getImageAndLabel(image, frame.camera_labels, [3, 3, indexImage+1], f)
                f.close
                plt.imsave("/content/drive/MyDrive/Waymo/training/images/" + name + ".jpg", tf.image.decode_jpeg(image.image).numpy(), cmap=None)

    return imageCount





if __name__ == "__main__":
    
    #Authentication at Google cloud
    # Requires Google Account with access to Waymo Open Dataset
    #auth.authenticate_user()

    segments = []
    imageCount = 0
        
    with open('/content/drive/MyDrive/Waymo/waymo-segments-training1', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            seg = ' '.join(row)
            segments.append(seg)


    for i, segment in enumerate(segments): 

    
        file_name = "/content/dataset/segment" + str(i) + ".tfrecord"

        #download the segment file from Google Cloud
        #gsutil cp gs://{bucket_name}/{segment} {file_name}

        imageCount += parseFile(file_name, i+399)
        

