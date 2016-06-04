# -*- coding: utf-8 -*-
import sys,os
import glob
import random
import itertools
DATASET_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/dataset'

def sampleEqualSubset(fpath, num_each, num_label = 2):
    lines = open(fpath,'r').readlines()
    random.shuffle(lines)
    print len(lines)
    assert num_label*num_each <= len(lines)
    imgSamples = [[] for _ in xrange(num_label)]
    for line in lines:
        sp = line[:-1].split()
        if len(imgSamples[int(sp[1])]) < num_each:
            imgSamples[int(sp[1])].append(sp[0])
    ret = imgSamples[0]
    for l in imgSamples[1:]:
        ret+=l
    return ret
if __name__ == '__main__':
    assert len(sys.argv)>2
    img = sampleEqualSubset(sys.argv[1],int(sys.argv[2]))
    print img

