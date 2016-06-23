# -*- coding: utf-8 -*-
import sys,os
import numpy as np

def us10kLoader(fpath, ipath = ''):
    lines = open(fpath,'r').readlines()
    labelList = lines[0].strip().split("\t")
    labelList = labelList[2:]
    print 'number of label',len(labelList)
    imList,dataList = [], []
    for l in lines[1:]:
        sp = l.strip().split("\t")
        imList.append(os.path.join(ipath,sp[0]))
        dataList.append([float(i) for i in sp[2:]])
        assert len(dataList[-1]) == len(labelList)
    return labelList,imList,np.asarray(dataList,dtype = 'float')
        

