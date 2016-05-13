# -*- coding: utf-8 -*-
#Simple annotation tool for single binary label (default) or multi-class label (up to 10 for now)
'''
usage: 
Dependency: OpenCV python bindings (pip install pyopencv)

Store all the images in a directory

To run: python simpleAnnotationTool.py directory_containing_images [file_name_of_annotation_data(default annotation.txt) number_of_labels(default 2) ]

Press digit key to annotate (1 for label 0 (e.g. male statue), 2 for label 1 (e.g. female statue), ...)

The annotation file will be stored in the same directory with the images

Press q to quit
'''
import sys,os
import cv2

def list_dir(pdir):
    ls = os.listdir(pdir)
    ret = [i for i in ls if os.path.isdir(os.path.join(pdir,i))]
    ret.sort()
    return ret

image_type = [".jpg"]

def list_images(dir):
    files = os.listdir(dir)
    img = [f for f in files if ( (os.path.splitext(f))[1] in image_type) and os.path.isfile(os.path.join(dir,f))]
    img.sort()
    return img

def list_all(pdir,sdir):
    return [list_images(os.path.join(pdir,d)) for d in sdir]


def key2label(key, numLabel = 2):
    lab = chr(key)
    if lab == 'q':
        return -2
    if not lab.isdigit():
        return -1
    lab = int(lab)
    lab = (lab-1 if lab>0 else 9)
    if lab>-1 and lab<numLabel:
        return lab
    else:
        return -1
    

NUM_LABEL = 2
labelFile = 'annotation.txt'
if __name__ == '__main__':
    assert len(sys.argv)>1
    basePath = sys.argv[1]
    if len(sys.argv)>2:
        labelFile = sys.argv[2]
    if len(sys.argv)>3:
        NUM_LABEL = int(sys.argv[3])
    images = list_images(basePath)
    cv2.namedWindow("Image to annotate")
    OF = open(os.path.join(basePath,labelFile),'w')
    OF.close()
    for imn in images:
        img = cv2.imread(os.path.join(basePath,imn))
        cv2.imshow('Image to annotate',img)
        label = -1
        while label < 0:
            pressed = cv2.waitKey(0)
            label = key2label(pressed)
            if label == -2:
                sys.exit()
            if label < 0:
                print 'key not valid'
        OF = open(os.path.join(basePath,labelFile),'a')
        print >>OF,imn,label
        OF.close()
           
