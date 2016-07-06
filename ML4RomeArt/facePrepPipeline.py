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

import cv2
import dlib
from skimage import io
from skimage import img_as_ubyte
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from FaceFrontalisation.facefrontal import getDefaultFrontalizer

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
#: Landmark indices corresponding to the inner eyes and bottom lip.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

#: Landmark indices corresponding to the outer eyes and nose.
OUTER_EYES_AND_NOSE = [36, 45, 33]
imgDim = 256
ENLARGE = 1.25

MODE = 1
FILTER_MODE = 1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(
            "Give the path to the trained shape predictor model as the first "
            "argument and then the directory containing the facial images.\n")
        exit()

    predictor_path = sys.argv[1]
    faces_folder_path = sys.argv[2]
    outputDir = os.path.join(faces_folder_path,'pre-process')
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    if os.path.isdir(outputDir):
        filterListFile = open(os.path.join(outputDir,'filtered_list.txt'),'w')
        if FILTER_MODE == 1:
            goodList = open(os.path.join(outputDir,'good_list.txt'),'w')
    else:
        print('Directory doesnt exists')
        exit()

    landmarkIndices=INNER_EYES_AND_BOTTOM_LIP

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()
    if MODE == 1:
        fronter = getDefaultFrontalizer()
    cv2.namedWindow('normalizedFace')
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        if len(img.shape) < 3:
            cimg = np.ndarray((img.shape[0],img.shape[1],3),dtype = 'uint8')
            for k in xrange(3):
                cimg[:,:,k] = img[:,:]
            img = cimg.copy()
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        if len(dets) < 1:
            print "No face detected, skipping"
            continue
        print("Number of faces detected: {}".format(len(dets)))
        maxArea = -1
        for k, d in enumerate(dets):
            if d.area() > maxArea:
                maxArea, maxD = d.area, d
        d = maxD
        print("Detection with max area: Left: {} Top: {} Right: {} Bottom: {}".format(
            d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        win.clear_overlay()
        win.set_image(img)
        shape = predictor(img, d)
        win.add_overlay(shape)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
        landmarks = list(map(lambda p: (p.x, p.y), shape.parts()))
        npLandmarks = np.float32(landmarks)
        if MODE == 0:
            npLandmarkIndices = np.array(landmarkIndices)            
            H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                       imgDim * MINMAX_TEMPLATE[npLandmarkIndices]+imgDim*(ENLARGE-1.0)/2)
            thumbnail = cv2.warpAffine(img, H, (int(imgDim*ENLARGE), int(imgDim*ENLARGE)))
            fnSurf = '_crop.jpg'
        else:
            thumbnail = fronter.frontalizeImage(img,d,npLandmarks)
            cut = thumbnail.shape[0]/8
            thumbnail = thumbnail[cut:thumbnail.shape[0]-cut,cut:thumbnail.shape[1]-cut,:].copy()
            fnSurf = '_frontal.jpg'
        cv2.imshow('normalizedFace',thumbnail)
        imPath, imName = os.path.split(f)
        if FILTER_MODE == 1:
            k = cv2.waitKey(-1)
            if k == 49:
                print 'good',imName
                print >> goodList, imName
        print >>filterListFile, imName,d.left(),d.top(),d.right(),d.bottom(),landmarks
        io.imsave(os.path.join(outputDir,imName+fnSurf),thumbnail)
        sys.stdout.flush()
        
