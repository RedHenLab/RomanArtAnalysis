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
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_extraction.text import CountVectorizer as HV
import glob
import pandas as pd
import re
from collections import Iterable

def collectJson(jpath):
    jfile = glob.glob(os.path.join(jpath,'*.json'))
    jList = [json.load(open(jf,'r')) for jf in jfile]
    newList = []
    for jd in jList:
        newd = {}
        for k in jd:
            newd[k.strip().strip(':')] = jd[k]
        newList.append(newd)
    df = pd.DataFrame(newList, index=[jf[jf.rfind('/')+1:-5] for jf in jfile])
    return df

TOKEN_PAT = r'\b\w\w*\-?\w+\b'

fieldMask1 = ['Attributes','Awarder','Beard','City','Clothing','Details','Earliest','Gender','Hairstyle', 'Honorand','Latest','Material', 'Object','Position', 'Province', 'Re-Use?', 'Region', 'Title']
fieldMask2 = ['Page_title','Title','label','Keywords']
key1 = [['male'],['female']]
key2 = [['male','man'],['female','woman']]
key3 = [['bearded','long-bearded','short-bearded'],['clean-shaven','shaven']]
key4 = [['long-bearded', 'long'],['short','short-bearded'],['stubble','stubble-bearded'],['clean-shaven','shaven']]
def buildWordHash(df):
    portId, portStr = [], []
    for po,rc in df.iterrows():
        portId.append(po)
        ks = []
        for k in rc:
            if isinstance(k, Iterable) and (type(k) != str):
                s = ' '.join([j for j in k])
            else:
                s= str(k)
            ks.append(s)
        portStr.append(' '.join(ks))
    hashVectorizer = HV(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words='english', token_pattern=TOKEN_PAT, ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=2, max_features=None, vocabulary=None, binary=True)
    wdCount = hashVectorizer.fit_transform(portStr)
    print 'word count matrix shape',wdCount.shape
    return portId,wdCount,hashVectorizer.vocabulary_

def loadFiltered(fileName):
    fpath,fname = os.path.split(fileName)
    lines = open(fileName,'r').readlines()
    filterd = {}
    for line in lines:
        filterd[line.split()[0]] = os.path.abspath(os.path.join(fpath,line.split()[0]+'_crop.jpg'))
    return filterd

def simpleTokenizer(ss):
    toknizer = re.compile(TOKEN_PAT)
    return toknizer.findall(ss.lower())

def buildFromField(df, imgList, filterd, field, key, endPat):
    dataset = []
    for po,rc in df.iterrows():
        token = simpleTokenizer(rc[field])
        label = -1
        for i,kg in enumerate(key):
            flag = False
            for k in kg:
                if k in token:
                    flag = True
                    break
            if flag:
                if label == -1:
                    label = i
                else:
                    label = -2
        if label >= 0:
            thisImg = [im for im in imgList if im[:im.find(endPat)] == po]
            dataset = dataset + [(filterd[im],label) for im in thisImg if im in filterd]
    return dataset

def buildFromWords(wdCount, portId, vocabulary, imgList, filterd, key, endPat):
    dataset = []
    for idx,po in enumerate(portId):
        label = -1
        for i,kg in enumerate(key):
            flag = False
            for k in kg:
                if wdCount[idx,vocabulary[k]] > 0:
                    flag = True
                    break
            if flag:
                if label == -1:
                    label = i
                else:
                    label = -2
        if label >= 0:
            thisImg = [im for im in imgList if im[:im.find(endPat)] == po]
            dataset = dataset + [(filterd[im],label) for im in thisImg if im in filterd]
    return dataset

def listImage(ipath):
    il = glob.glob(os.path.join(ipath,'*.jpg'))
    il = [os.path.split(i)[1] for i in il]
    return il

if __name__ == '__main__':
    assert len(sys.argv)>=4
    totalDF = collectJson(sys.argv[1])
    filterd = loadFiltered(sys.argv[2])
    imgList = listImage(sys.argv[1])
    print 'all the columns we have',totalDF.columns
    print 'the number of rows (subjects)',len(totalDF.index)
    #newDf  = totalDF[fieldMask2]
    newDf  = totalDF[fieldMask1]
    portId,wdCount,vocabulary = buildWordHash(newDf)
   
    #data = buildFromField(newDf,imgList, filterd, 'Gender',key1, '_')
    data = buildFromField(newDf,imgList, filterd, 'Beard',key4, '_')
    #data = buildFromWords(wdCount,portId, vocabulary, imgList, filterd, key2,'.jpg')
    print len(data)
    OF = open(sys.argv[3],'w')
    for line in data:
        print >>OF,line[0],line[1]
    OF.close
