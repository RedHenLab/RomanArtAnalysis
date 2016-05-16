# -*- coding: utf-8 -*-
#Simple annotation tool for single binary label (default) or multi-class label (up to 10 for now)
usage = '''
usage: 
Dependency: OpenCV python bindings (pip install pyopencv)

Store all the images in a single directory

To run: python simpleAnnotationTool.py Path_to_the_directory_containing_images Path_to_the_parameter_file 

The parameter file should look like:

key1 tag1
key2 tag2
...

e.g.

1 male
2 female
3 notSure

The keys can be simple characters on the key board like 0,1,2,a,b,c, but q can not be used.


Press key to annotate (map from keys to tags is in the parameter file)

press space to skip current image

The annotation file will be stored in the same directory with the images

The file name is annotation__numberOfLabel-label0-label1-...-lastLabel__year-month-day_hh-mm-ss__randomNumber.txt

e.g. annotation__3-male-female-notSure__2016-05-16_19-40-06__54.txt

The random number from 0 to 99 is to avoid possible conflict

Press q to quit
'''
import sys,os
import cv2
import time
import random

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


def key2label(key, keyMap):
    if key>255:
        return -1
    lab = chr(key)
    if lab == ' ':#press space will skip
        return -3
    if lab == 'q':#press q will quit
        return -2 
    if not lab.isdigit():
        return -1
    if lab not in keyMap:
        return -1
    else:
        return lab
    
def parseKey(keyFile):
    lines = open(keyFile,'r').readlines()
    num = len(lines)
    desc = str(num)+'-'
    kmap = {}
    for line in lines:
        sp = line.strip().split()
        kmap[sp[0]] = sp[1]
        desc = desc+sp[1]+'-'
    return kmap,desc[:-1],num

NUM_LABEL = 2
labelFile = 'annotation'
if __name__ == '__main__':
    assert len(sys.argv)>2,usage
    basePath = sys.argv[1]
    keyFile = sys.argv[2]
    keyMap,keyDesc,NUM_LABEL = parseKey(keyFile)
    labelFile = labelFile+'__'+keyDesc+'__'+time.strftime('%Y-%m-%d_%H-%M-%S')+'__'+str(random.randint(0,100))+'.txt'
    #file name: prefix__numLabel-label0-label1-...-lastLabel__year-month-day_hh-mm-ss__randomNumber.txt
    images = list_images(basePath)
    cv2.namedWindow("Image to annotate")
    OF = open(os.path.join(basePath,labelFile),'w')
    print >>OF, '#Image folder is',basePath
    print >>OF, '#parameter file is',keyFile
    OF.close()
    for imn in images:
        img = cv2.imread(os.path.join(basePath,imn))
        cv2.imshow('Image to annotate',img)
        label = -1
        while label < 0:
            pressed = cv2.waitKey(0)
            label = key2label(pressed,keyMap)
            if label == -2:
                sys.exit()
            if label == -3:
                break
            if label < 0:
                print 'key not valid'
        if label <0:
            continue #skip this image
        OF = open(os.path.join(basePath,labelFile),'a')
        print >>OF,imn,keyMap[label]
        OF.close()
           
