# -*- coding: utf-8 -*-
import simplejson as json
import sys,os
import numpy as np
import cPickle as pickle
import gzip
import sklearn
from sklearn import preprocessing as prep
from sklearn import cross_validation as cross
from sklearn import metrics as met
from sklearn.pipeline import Pipeline
import glob
import pandas as pd
import random
import cv2
import dlib
from skimage import io

imgDim = 256
ENLARGE = 1.25
SAMPLE_RATE = 0.1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(
            "Give the path to the directory containing the facial images.\n")
        exit()

    #predictor_path = sys.argv[1]
    faces_folder_path = sys.argv[1]
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(predictor_path)
    imagesAll = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    sampleNum = int(len(imagesAll)*SAMPLE_RATE)
    print 'total image number',len(imagesAll),'sample',sampleNum,'images for test'
    imagesSample = random.sample(imagesAll,sampleNum)
    countMat = [[0,0],[0,0,0]]
    win = dlib.image_window()
    cv2.namedWindow('img')
    for f in imagesSample:
        print("Processing file: {}".format(f))
        img = io.imread(f)
        imgcv = cv2.imread(f)
        win.clear_overlay()
        win.set_image(img)
        if imgcv is None or imgcv.shape[0]<=0 or imgcv.shape[1]<=0:
            continue
        cv2.imshow('img',imgcv)
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        if len(dets) < 1:
            print "No face detected, skipping"
            isdet = 0
        else:
            isdet = 1
            print("Number of faces detected: {}".format(len(dets)))
        win.add_overlay(dets)
        #dlib.hit_enter_to_continue()
        key = chr(cv2.waitKey(0))
        while not (key == '1' or key == '2' or key == '3' or key == 'q'):
            key = chr(cv2.waitKey(0))
        print key
        if key == '1':
            ispres = int(key)-1
            countMat[ispres][isdet]+=1
        else:
            if isdet<1:
                countMat[1][0]+=1
            else:
                countMat[1][int(key)-1]+=1
    print countMat
        
'''
for ancientrome.ru, the mat is [[58,6],[44,139,0]], detection recall is 75.96%
for laststatues, the mat is [[195,0],[37,94,0]], detection recall is 71.76%
'''
