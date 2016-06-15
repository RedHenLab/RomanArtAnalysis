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
from sklearn.svm import SVR
import glob
import pandas as pd
import scipy.io as sio
import math
from sklearn.grid_search import GridSearchCV
import shutil 
labelList = ['Old', 'Masculine', 'Baby-faced', 'Competent', 'Attractive', 'Energetic', \
             'Well-groomed', 'Intelligent', 'Honest', 'Generous', \
             'Trustworthy', 'Confident', 'Rich', 'Dominant']

MODEL_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/models/'
MODEL_PREFIX = 'social_landmarks_SVR_'
RESULT_PATH = 'keywordResults/'
ARG_LEN = 10

def median_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true))

def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def almost_correct_value_percentage(y_true, y_pred, diff = 0.1):
    ct = 0
    for i in xrange(y_true.shape[0]):
        if np.abs( (y_true[i]-y_pred[i])/y_true[i] ) <= diff:
            ct += 1
    return float(ct)/y_true.shape[0]

def almost_correct_rank_percentage(y_true, y_pred, diff = 10):
    ct = 0
    pos_true = [0 for _ in xrange(y_true.shape[0])]
    pos_pred = [0 for _ in xrange(y_true.shape[0])]
    for pos,idx in enumerate(np.argsort(y_true)):
        pos_true[idx] = pos
    for pos,idx in enumerate(np.argsort(y_pred)):
        pos_pred[idx] = pos
    for i in xrange(y_true.shape[0]):
        if np.abs(pos_true[i]-pos_pred[i]) <= diff:
            ct += 1
    return float(ct)/y_true.shape[0]

def customScorer(estimator, X,y):
    return -median_absolute_percentage_error(y,estimator.predict(X))
     
def filterNaN(imName,dataX):
    flags = [True for _ in xrange(dataX.shape[0])]
    for i in xrange(dataX.shape[0]):
        for j in xrange(dataX.shape[1]):
            if math.isnan(dataX[i,j]) or abs(dataX[i,j]) > 1e9:
                print 'encounter NaN',i,j
                flags[i] = False
                break
    imName = [n for f,n in zip(flags,imName) if f]
    dataX = np.asarray([dataX[i,:] for i,f in enumerate(flags) if f],dtype = 'float')
    dataX = dataX.reshape((dataX.shape[0],-1))
    return imName,dataX

if __name__ == '__main__':
    assert len(sys.argv)>1
    if sys.argv[1] == 'train':
        assert len(sys.argv[2:]) >= 2
        dataY = sio.loadmat(sys.argv[2])['trait_annotation']
        imName,dataX = pickle.load(open(sys.argv[3],'rb'))
        print np.max(dataX),np.min(dataX)
        assert dataX.shape[0] == dataY.shape[0]
        dataX = prep.minmax_scale(dataX)
        print np.max(dataX),np.min(dataX)
        baseReg = SVR(kernel='linear', gamma=0.1, coef0=0.0, C=1.0, epsilon=0.2, verbose=False)
        numLabel = dataY.shape[1]
        wmscore = 0.0
        paramGridLinear = {'C':(0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('linear',),'epsilon':(0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.8,1.0)}
        paramGridLinearC = {'C':(0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0,1000.0),'kernel':('linear',),'epsilon':(0.01,0.05,0.1,0.2,0.3)}
        paramGridRbf = {'gamma':(0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1.0),'C':(0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('rbf',),'epsilon':(0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.8,1.0)}
        paramGridRbfC = {'gamma':(0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0),'C':(0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('rbf',),'epsilon':(0.01,0.05,0.1,0.2,0.3)}
        
        #paramGridAll = [paramGridLinear,paramGridRbf]
        paramGridAll = [paramGridLinearC]
        for l in xrange(1,numLabel):
            print np.min(dataY[:,l]),np.max(dataY[:,l]),np.max(dataY[:,l])-np.min(dataY[:,l]),np.std(dataY[:,l])
            gscv = GridSearchCV(baseReg,paramGridAll, scoring = customScorer,cv = 10, n_jobs=8, refit = True,verbose = 1)
            gscv.fit(dataX, dataY[:,l])
            #score = cross.cross_val_score(baseReg, dataX, dataY[:,l], scoring = 'mean_squared_error',cv = 10, n_jobs=8)
            score = gscv.best_score_
            print gscv.best_params_
            print 'score for social attr',l,labelList[l],-np.mean(score),'+-',np.std(score)
            wmscore += -np.mean(score)
            savePath = os.path.join(MODEL_PATH,MODEL_PREFIX+str(l)+'_'+labelList[l]+'.pkl')
            OF = open(savePath,'wb')
            pickle.dump(gscv,OF,-1)
            OF.close()
            print 'model saved to',savePath
            sys.stdout.flush()
        print wmscore/numLabel
        
    elif sys.argv[1] == 'validation':
        assert len(sys.argv[2:]) >= 2
        dataY = sio.loadmat(sys.argv[2])['trait_annotation']
        imName,dataX = pickle.load(open(sys.argv[3],'rb'))
        assert dataX.shape[0] == dataY.shape[0]
        numLabel = dataY.shape[1]
        for l in xrange(numLabel):
            savePath = os.path.join(MODEL_PATH,MODEL_PREFIX+str(l)+'_'+labelList[l]+'.pkl')
            rgr = pickle.load(open(savePath,'rb'))
            pred = rgr.predict(dataX)
            argPredict = np.argsort(pred)
            argTrue = np.argsort(dataY[:,l])
            print 'attribute',labelList[l]
            print 'explained_variance_score',met.explained_variance_score(dataY[:,l],pred)
            print 'r2_score',met.r2_score(dataY[:,l],pred)
            print 'mean_absolute_error',met.mean_absolute_error(dataY[:,l],pred)
            print 'median_absolute_error',met.median_absolute_error(dataY[:,l],pred)
            print 'mean_absolute_percentage_error',mean_absolute_percentage_error(dataY[:,l],pred)
            print 'median_absolute_percentage_error',median_absolute_percentage_error(dataY[:,l],pred)
            print 'almost_correct_value_percentage',almost_correct_value_percentage(dataY[:,l],pred,0.1)
            print 'almost_correct_rank_percentage',almost_correct_rank_percentage(dataY[:,l],pred,10)
            #print argPredict[:20],argPredict[-20:]
            #print argTrue[:20],argTrue[-20:]
            ct = 0
            for i in argPredict[:20]:
                if i in argTrue[:20]:
                    ct+=1
            print 'least ones hit',ct
            ct = 0
            for i in argPredict[-20:]:
                if i in argTrue[-20:]:
                    ct+=1
            print 'most ones hit',ct
            
            
    elif sys.argv[1] == 'transfer':
        assert len(sys.argv[2:]) >= 2
        imName,dataX = pickle.load(open(sys.argv[2],'rb'))
        imName,dataX = filterNaN(imName,dataX)
        dataX = prep.robust_scale(dataX)
        xlen,numLabel = dataX.shape[0],len(labelList)
        socialEval = np.ndarray((xlen,numLabel),dtype = 'float')
        for l in xrange(numLabel):
            savePath = os.path.join(MODEL_PATH,MODEL_PREFIX+str(l)+'_'+labelList[l]+'.pkl')
            sreg = pickle.load(open(savePath,'rb'))
            socialEval[:,l]= sreg.predict(dataX)
        print 'social evaluation results for',sys.argv[2],'saved to',sys.argv[3]
        OF = open(sys.argv[3],'wb')
        pickle.dump((imName,socialEval),OF,-1)
        OF.close()
        
    elif sys.argv[1] == 'analysis':
        assert len(sys.argv[2:]) >= 2
        numLabel = len(labelList)
        portId,stats, wdCount = pickle.load(open(sys.argv[2],'rb'))
        imName,socialEval = pickle.load(open(sys.argv[3],'rb'))
        imId = []
        for n in imName:
            f = os.path.split(n)[1]
            imId.append(f[:f.rfind('_')])
        print 'unique objects who have face evaluation', len(set(imId))
        ohash = {}
        for i,oid in enumerate(imId):
            if oid not in ohash:
                ohash[oid] = [socialEval[i,:],1]
            else:
                ohash[oid][0] += socialEval[i,:]
                ohash[oid][1] += 1
        for oid in ohash:
            ohash[oid][0]/=ohash[oid][1]
        commonID, commonWC, commonEval = [], [], []
        for i,oid in enumerate(portId):
            if oid in ohash:
                commonID.append(oid)
                commonWC.append(wdCount[i,:])
                commonEval.append(ohash[oid][0])
        commonWC = np.asarray(commonWC)
        commonWC.reshape((commonWC.shape[0],-1))
        commonEval = np.asarray(commonEval)
        commonEval.reshape((commonEval.shape[0],-1))
        commonData = np.concatenate([commonWC,commonEval], axis = 1)
        corr = np.corrcoef(commonData,rowvar = 0)[:,-numLabel:]
        stats += [[l,1] for l in labelList]
        OF = open(RESULT_PATH+'correlationKeywordsSocialEval.csv','w')
        tstr = ''
        for s in labelList:
            tstr+=','+s
        print >>OF,tstr
        for i in xrange(corr.shape[0]):
            tstr = str(stats[i][0])+','
            for j in xrange(corr.shape[1]):
                tstr= tstr+"{:.4f}".format(corr[i,j])+','
            print>>OF,tstr[:-1]
        OF.close()

        corrPairs = []
        for i in xrange(corr.shape[0]-numLabel):
            for j in xrange(corr.shape[1]):
                if not math.isnan(corr[i,j]):
                    corrPairs.append((stats[i][0], labelList[j], corr[i,j]))
        corrPairs.sort(key = lambda x:abs(x[2]))
        corrPairs.reverse()

        OF = open(RESULT_PATH+'keywordSocialCorrPairsSorted.csv','w')
        for k,s,c in corrPairs:
            tstr = k+','+s+','+ "{:.4f}".format(c)
            print>>OF,tstr
        OF.close()
    elif sys.argv[1] == 'argsort':
        assert len(sys.argv[2:]) >= 3,'usage: pickled social evaluation results, outputpath, croped face path'
        imName,socialEval = pickle.load(open(sys.argv[2],'rb'))
        outputPath = sys.argv[3]
        facePath = sys.argv[4]
        def copyFaceImg(idx,surf,lab):
            fname = os.path.split(imName[idx])[1]
            opath = os.path.join(facePath,fname+'_crop.jpg')
            if not os.path.isfile(opath):
                print opath,'is not a file'
                return
            if surf == 'most':
                oid = abs(idx)
            else:
                oid = idx+1
            dpath = outputPath+'_'+surf+'_'+l+'_'+str(oid)+'.jpg'
            shutil.copy(opath,dpath)
            
        for i,l in enumerate(labelList):
            asort = np.argsort(socialEval[:,i])
            for n in xrange(ARG_LEN):
                copyFaceImg(n,'least',l)
                copyFaceImg(-(n+1),'most',l)
    elif sys.argv[1] == 'compare':
        pass
    else:
        print 'please use train or transfer or analysis command'
    
