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

def countCoocc(countMat):
    res = np.zeros((countMat.shape[1],countMat.shape[1]),dtype = 'int')
    for i in xrange(countMat.shape[0]):
        for j in xrange(countMat.shape[1]):
            if countMat[i,j] > 0:
                for k in xrange(j+1,countMat.shape[1]):
                    if countMat[i,k] > 0:
                        res[j,k]+=1
                        res[k,j]+=1
    return res


RESULT_PATH = 'keywordResults/'
MAX_NUM = 20
if __name__ == '__main__':
    totalDF = collectJson(sys.argv[1])
    print 'all the columns we have',totalDF.columns
    print 'the number of rows (subjects)',len(totalDF.index)
    newDf  = totalDF[['Attributes','Awarder','Beard','City','Clothing','Details','Earliest','Gender','Hairstyle', 'Honorand','Latest','Material', 'Object','Position', 'Province', 'Re-Use?', 'Region', 'Title']]
    portId, portStr = [], []
    for po,rc in newDf.iterrows():
        portId.append(po)
        portStr.append(' '.join([k for k in rc]))
    hashVectorizer = HV(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words='english', token_pattern=r'\b\w\w*\-?\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=2, max_features=None, vocabulary=None, binary=True)
    wdCount = hashVectorizer.fit_transform(portStr)
    print 'word count matrix shape',wdCount.shape
    occs = wdCount.sum(axis = 0)
    argOC = np.argsort(occs)
    wdMap = {}
    for k in hashVectorizer.vocabulary_:
        wdMap[hashVectorizer.vocabulary_[k]]=k
    stats = [(wdMap[argOC[0,i]],occs[0,argOC[0,i]]) for i in xrange(occs.shape[1])]
    stats.reverse()
    sortedWordCount = wdCount.toarray()
    for ct, s in enumerate(stats):
        idx = hashVectorizer.vocabulary_[s[0]]
        sortedWordCount[:,ct] = wdCount[:,idx].toarray().flatten()
    corr = np.corrcoef(sortedWordCount,rowvar = 0)
    cooccs = countCoocc(sortedWordCount)
    absCorr = np.abs(corr)
    #print corr.shape
    corrMax = []
    absCorrMax = []
    for i in xrange(corr.shape[0]):
        ars = list(np.argsort(corr[i,:])[-MAX_NUM:])
        ars.reverse()
        corrMax.append(ars)
        ars = list(np.argsort(absCorr[i,:])[-MAX_NUM:])
        ars.reverse()
        absCorrMax.append(ars)

    OF = open(RESULT_PATH+'wordCount.csv','w')
    for s in stats:
        print>>OF, s[0]+','+str(s[1])
    OF.close()

    OF = open(RESULT_PATH+'cooccurrence.csv','w')
    tstr = ''
    for s in stats:
        tstr+=','+s[0]
    print >>OF,tstr
    for i in xrange(cooccs.shape[0]):
        tstr = stats[i][0]+','
        for j in xrange(cooccs.shape[1]):
            tstr= tstr+str(cooccs[i,j])+','
        print>>OF,tstr[:-1]
    OF.close()

    OF = open(RESULT_PATH+'correlationCoef.csv','w')
    tstr = ''
    for s in stats:
        tstr+=','+s[0]
    print >>OF,tstr
    for i in xrange(corr.shape[0]):
        tstr = str(stats[i][0])+','
        for j in xrange(corr.shape[1]):
            tstr= tstr+"{:.4f}".format(corr[i,j])+','
        print>>OF,tstr[:-1]
    OF.close()

    OF = open(RESULT_PATH+'correlationSort.csv','w')
    for i in xrange(len(corrMax)):
        tstr = stats[i][0]+','
        for idx in corrMax[i][1:]:
            tstr = tstr+ stats[idx][0]+','+"{:.4f}".format(corr[i,idx])+','
        print>>OF,tstr[:-1]
    OF.close()

    OF = open(RESULT_PATH+'absoluteCorrelationSort.csv','w')
    for i in xrange(len(absCorrMax)):
        tstr = stats[i][0]+','
        for idx in absCorrMax[i][1:]:
            tstr = tstr+ stats[idx][0]+','+"{:.4f}".format(corr[i,idx])+','
        print>>OF,tstr[:-1]
    OF.close()

