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
import math

import cv2
import dlib
from skimage import io

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from FaceFrontalisation.facefrontal import getDefaultFrontalizer

MODE = 1

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
predictor_path = "/home/mfs6174/shape_predictor_68_face_landmarks.dat"
landmarkIndices=INNER_EYES_AND_BOTTOM_LIP

def getCos(x,y):
    lx,ly = np.sqrt(np.dot(x,x)),np.sqrt(np.dot(y,y))
    return np.dot(x,y)/(lx*ly)
def getEccentricity(p0,p1,p2,p3 = None):
    a = edis(p0,p1)/2.0
    if p3 is None:
        p3 = p2
    b = abs((p0[1]+p1[1])/2.0-(p2[1]+p3[1])/2.0)
    if b > a:
        b = a
    return np.sqrt(a**2-b**2)/a
def edis(a,b):
    return np.sqrt(np.dot(a-b,a-b))

def imgEnhance(img):
    blur = cv2.GaussianBlur(img,(0,0),20)
    img = cv2.addWeighted(img,1.1,blur,-0.1,0)
    #return blur
    return img

def getNormalizedLandmarks(img, predictor, d, fronter = None, win2 = None):
    shape = predictor(img, d)
    landmarks = list(map(lambda p: (p.x, p.y), shape.parts()))
    npLandmarks = np.float32(landmarks)
    if MODE == 0:
        npLandmarkIndices = np.array(landmarkIndices)            
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                MINMAX_TEMPLATE[npLandmarkIndices])
        normLM = cv2.transform(np.asarray([npLandmarks]),H)[0,:,:]
        return normLM,shape
    else:
        assert fronter is not None
        thumbnail = fronter.frontalizeImage(img,d,npLandmarks)
        #thumbnail = imgEnhance(thumbnail)
        cut = thumbnail.shape[0]/5
        thumbnail = thumbnail[cut+5:thumbnail.shape[0]-cut-5,cut+10:thumbnail.shape[1]-cut-10,:].copy()
        newShape = predictor(thumbnail, dlib.rectangle(0,0,thumbnail.shape[0],thumbnail.shape[1]))
        if win2 is not None:
            win2.clear_overlay()
            win2.set_image(thumbnail)
            win2.add_overlay(newShape)
            #dlib.hit_enter_to_continue()
        landmarks = list(map(lambda p: (float(p.x)/thumbnail.shape[0], float(p.y)/thumbnail.shape[1]), newShape.parts()))
        npLandmarks = np.float32(landmarks)
        normLM = npLandmarks
    return normLM,shape
        
        

def extractFaceFeature(img, predictor, d, fronter = None, win2 = None):
    ret = getNormalizedLandmarks(img, predictor, d, fronter, win2)
    if ret == False:
        return False
    normLM,shape = ret
    #length group
    lew = edis(normLM[36],normLM[39]) #left eye width
    rew = edis(normLM[42],normLM[45]) #right eye width
    leh = (edis(normLM[37],normLM[41])+edis(normLM[38],normLM[40]))/2.0#left eye height
    reh = (edis(normLM[43],normLM[47])+edis(normLM[44],normLM[46]))/2.0#right eye height
    mew = max(lew,rew)#max eye width
    meh = max(leh,reh)#max eye height
    noh = edis(normLM[27],normLM[33]) #nose height
    now = edis(normLM[31],normLM[35]) #nose width
    #face width1
    fw1 = edis(normLM[0],normLM[16])
    fw2 = edis(normLM[2],normLM[14])
    fw3 = edis(normLM[4],normLM[12])
    fw4 = edis(normLM[6],normLM[10])
    #face length
    flen = abs((normLM[0][1]-normLM[16][1])/2.0-normLM[8][1])
    #lip thick
    upliph =(edis(normLM[50],normLM[61]) + edis(normLM[52],normLM[63]))/2.0
    btliph = (edis(normLM[67],normLM[58]) + edis(normLM[66],normLM[57]) + edis(normLM[65],normLM[56]) )/3.0
    #mouth width
    mouw = edis(normLM[48],normLM[54])
    
    #ratio group
    _den = abs( (normLM[37][1]+normLM[38][1])/2.0-normLM[33][1] )
    l1 = abs( (normLM[37][1]+normLM[38][1])/2.0-normLM[19][1] ) / _den
    l2 = abs(normLM[51][1] - normLM[33][1] ) / _den
    l3 = abs(normLM[57][1] - normLM[33][1] ) / _den
    fr1 = fw4/fw3
    fr2 = fw4/fw2
    fr3 = fw4/fw1
    fr4 = fw3/fw2
    fr5 = fw3/fw1
    fr6 = fw2/fw1
    fr7 = fw1/flen
    eyer = mew/meh
    mour = (upliph+btliph)/mouw
    #angle cos group
    leba = getCos(normLM[17]-normLM[19], normLM[21]-normLM[19])#left eyebrow
    reba = getCos(normLM[22]-normLM[24], normLM[26]-normLM[24])#right eyebrow
    nora = getCos(normLM[31]-normLM[30], normLM[35]-normLM[30]) #nose root
    nota = getCos(normLM[31]-normLM[33], normLM[35]-normLM[33]) #nose tip
    china1 = getCos(normLM[4]-normLM[8], normLM[12]-normLM[8]) #chin
    lfa = getCos(normLM[0]-normLM[4], normLM[8]-normLM[4]) #left face
    rfa = getCos(normLM[16]-normLM[12], normLM[8]-normLM[12]) #right face
    moucora = getCos(normLM[60]-normLM[62], normLM[64]-normLM[62]) #mouth corrnor
    #eccentricity group
    e1 = getEccentricity(normLM[48],normLM[54],normLM[51])#upper mouth
    e2 = getEccentricity(normLM[48],normLM[54],normLM[57])#lower mouth
    e3 = getEccentricity(normLM[36],normLM[39],normLM[37],normLM[38])#upper left eye
    e4 = getEccentricity(normLM[36],normLM[39],normLM[40],normLM[41])#lower left eye
    e5 = getEccentricity(normLM[42],normLM[45],normLM[43],normLM[44])#upper right eye
    e6 = getEccentricity(normLM[42],normLM[45],normLM[46],normLM[47])#lower rigth eye
    e7 = getEccentricity(normLM[17],normLM[21],normLM[19])#left eye brown
    e8 = getEccentricity(normLM[22],normLM[26],normLM[24])#right eye brown
    '''features = np.asarray([lew,rew,leh,reh,mew,meh,noh,now,fw1,fw2,fw3,fw4,flen,upliph,btliph,mouw,\
                l1,l2,l2,fr1,fr3,fr4,fr5,fr6,fr7,eyer,mour,leba,reba,nora,nota,china1,lfa,rfa,\
                           moucora, e1,e2,e3,e4,e5,e6,e7,e8],dtype = 'float')'''
    features = np.asarray([lew,rew,leh,reh,mew,meh,noh,now,fw1,fw2,fw3,fw4,flen,upliph,btliph,mouw,\
                           leba,reba,nora,nota,china1,lfa,rfa,\
                           moucora, e1,e2,e3,e4,e5,e6,e7,e8],dtype = 'float')
    features = features.reshape((1,features.shape[0]))
    points  = normLM.reshape((1,-1))
    fullFeatures = np.concatenate((features, points), axis = 1)
    return fullFeatures,shape
    
if __name__ == '__main__':

    faces_folder_path = sys.argv[1]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    imNames = []
    features = []
    win = dlib.image_window()
    win2 = dlib.image_window()
    flist = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    flist.sort()
    fronter = getDefaultFrontalizer()
    for n,f in enumerate(flist):
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
        win.clear_overlay()
        win.set_image(img)
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
        ret = extractFaceFeature(img,predictor,d, fronter, win2)
        if ret == False:
            continue
        ft,shape = ret
        win.add_overlay(shape)
        dropFlag = False
        #dlib.hit_enter_to_continue()
        for i in xrange(ft.shape[1]):
            if math.isnan(ft[0,i]):
                print i
                dropFlag = True
                print 'drop',f
                dlib.hit_enter_to_continue()
                break
        if not dropFlag:
            imNames.append(f)
            features.append(ft)
            
    features = np.asarray(features, dtype = 'float')
    features = features.reshape((features.shape[0],features.shape[2]))
    OUTF = open(sys.argv[2],'wb')
    pickle.dump((imNames,features),OUTF,-1)
    OUTF.close()
        
