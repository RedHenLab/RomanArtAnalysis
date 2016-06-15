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
from sklearn.svm import SVR,SVC
import glob
import pandas as pd
import scipy.io as sio
import math
from sklearn.grid_search import GridSearchCV,ParameterGrid
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.base import clone,BaseEstimator
from buildDatasetFromKeywords import collectJson,buildWordHash
from CNN4Portraits_keras import copy_selected_po
NFOLDS = 10
random_state = 6174
MODEL_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/models/keywordBinaryModels'
KEYWORD_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/Code/ML4RomeArt/keywordResults'
fieldMask1 = ['Attributes','Awarder','Beard','City','Clothing','Details','Earliest','Gender','Hairstyle', 'Honorand','Latest','Material', 'Object','Position', 'Province', 'Re-Use?', 'Region', 'Title']
fieldMask2 = ['Page_title','Title','label','Keywords']

fieldMask = [fieldMask1, fieldMask2]
paramGridLinear = {'C':(0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('linear',)}
paramGridRbf = {'gamma':(0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1.0),'C':(0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('rbf',)}
PARA_GRID = [paramGridLinear,paramGridRbf]
#PARA_GRID = [paramGridLinear]


class EasyEnsemble(BaseEstimator):
    def __init__(self,baseEst, maxClf = None, verbose = False):
        self.baseEst = baseEst
        self.estList = []
        self.maxClf = maxClf
        self.verbose = verbose

    def predict_proba(self,X):
        assert len(self.estList)>0
        ap = None
        for est in self.estList:
            if ap is None:
                ap = est.predict_proba(X)
            else:
                ap = ap+est.predict_proba(X)
        ap = ap/len(self.estList)
        return ap
    def predict(self,X):
        prob = predict_proba(X)
        return np.argmax(prob,axia = 1)

    def fit(self,X,y):
        y = np.asarray(y, dtype = 'int')
        indexbyLab = [[],[]]
        for i,l in enumerate(y):
            assert l==0 or l==1,'EasyEnsemble only supports binary classification for now'
            indexbyLab[l].append(i)
        if len(indexbyLab[0])>len(indexbyLab[1]):
            majC = 0
            majN = len(indexbyLab[0])
        else:
            majC = 1
            majN = len(indexbyLab[1])
        if self.verbose:
            print 'major class is',majC
            print 'major sample number',majN,'minor sample number',len(indexbyLab[1-majC])
        nClf = majN/len(indexbyLab[1-majC])
        if self.maxClf is not None:
            nClf = min(nClf, maxClf)
        if self.verbose:
            print 'using',nClf,'meta classifiers for bagging'
        self.estList = [clone(self.baseEst) for n in xrange(nClf)]
        for n in xrange(nClf):
            #if self.verbose:
            #    print 'fitting classifier',n+1
            majIdx = np.random.choice(indexbyLab[majC],majN/nClf,replace = True)
            resampleX = np.concatenate((X[majIdx,:],X[indexbyLab[1-majC],:]), axis = 0)
            resampleY = np.concatenate((y[majIdx],y[indexbyLab[1-majC]]), axis = 0)
            self.estList[n].fit(resampleX,resampleY)

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'usage: pickled_Word_Count number_of_database face_feature json_path end_pattern'
    portId,stats,wdCount = pickle.load(open(sys.argv[1],'rb'))
    print 'valid words number',len(stats)
    dataY = {}
    for w,oc in stats:
      dataY[w] = []
    dataX = None
    idX = []
    assert len(sys.argv[2:]) > 1
    numDB = int(sys.argv[2])
    assert len(sys.argv[3:]) >= numDB*3
    imName, imFeature = [], []
    for i in xrange(0,numDB*3,3):
        iname,ift = pickle.load(open(sys.argv[3+i],'rb'))
        print ift.shape
        if dataX is None:
            dataX = ift
        else:
            dataX = np.concatenate([dataX,ift], axis = 0)
        print dataX.shape
        imName.append(iname)
        imFeature.append(ift)
        df = collectJson(sys.argv[3+i+1])
        df  = df[fieldMask[i]]
        pid,wc,voc = buildWordHash(df)
        endPat = sys.argv[3+i+2]
        imids = []
        for i,n in enumerate(iname):
            name = os.path.split(n)[1]
            oid = name[:name.find(endPat)]
            imids.append(oid)
        idX += imids
        pidHash = {}
        for i,p in enumerate(pid):
            pidHash[p] = i
        for w,oc in stats:
            dataY[w]+=[0 for _ in xrange(len(imids))]
            if w not in voc:
                continue
            widx = voc[w]
            for i,imid in enumerate(imids):
                if imid not in pidHash:
                    continue
                idx = pidHash[imid]
                if wc[idx,widx] > 0:
                    dataY[w][i] = 1

    uniqueObj = list(set(idX))
    PG = ParameterGrid(PARA_GRID)
    clfTemp = SVC(C=1.0, kernel='linear', probability=True)
    train_data = dataX
    po_id = idX
    OF1 = open(os.path.join(KEYWORD_PATH,'keywordClassifiersROC.csv'),'w')
    OF2 = open(os.path.join(KEYWORD_PATH,'keywordClassifiersROCGood.csv'),'w')
    for w in dataY:
        train_target = dataY[w]
        print 'training classifier for keyword:',w
        bestScore = -1
        posObj, negObj = set(),set()
        for oid,y in zip(idX,train_target):
            if y==0:
                negObj.add(oid)
            else:
                posObj.add(oid)
        posObj,negObj = list(posObj),list(negObj)
        minObj = min(len(posObj),len(negObj))
        if minObj < NFOLDS:
            print 'only',minObj,'objects with faces have','keyword',w,'skipping'
            continue
        nf = NFOLDS
        #print 'cross validation on folds number',nf
        kfp = KFold(len(posObj), n_folds=nf,
                    shuffle=True, random_state=random_state)
        kfn = KFold(len(negObj), n_folds=nf,
                    shuffle=True, random_state=random_state)
        for pg in PG:
            #print 'testing',pg
            ascore = []
            nfold = 0
            for (train_po, test_po),(train_ne,test_ne) in zip(kfp,kfn):
                nfold+=1
                train_dl_po = [posObj[train_po[i]] for i in xrange(len(train_po)) ]
                test_dl_po = [posObj[test_po[i]] for i in xrange(len(test_po)) ]
                train_dl_ne = [negObj[train_ne[i]] for i in xrange(len(train_ne)) ]
                test_dl_ne = [negObj[test_ne[i]] for i in xrange(len(test_ne)) ]
                train_dl = train_dl_po + train_dl_ne
                test_dl = test_dl_po + test_dl_ne
                train_data_d, train_target_d, trainIdx = copy_selected_po(train_data, train_target, po_id, train_dl)
                val_data, val_target, valIdx = copy_selected_po(train_data, train_target, po_id, test_dl)
                baseClf = clone(clfTemp).set_params(**pg)
                clf = EasyEnsemble(baseClf, verbose = False)
                clf.fit(train_data_d,train_target_d)
                prob = clf.predict_proba(val_data)
                score = met.roc_auc_score(val_target,prob[:,1])
                #print 'fold',nfold,'roc score',score
                ascore.append(score)
            avg = np.mean(ascore)
            stderr = np.std(ascore)
            #print 'average score cross folds',avg,'+-',stderr
            
            if avg > bestScore:
                bestScore = avg
                bestStd = stderr
                bestParam = pg
        print 'best socre is', bestScore, '+-', bestStd
        print 'best parameter is',bestParam
        print 'refitting'
        baseClf = clone(clfTemp).set_params(**bestParam)
        clf = EasyEnsemble(baseClf, verbose = True)
        clf.fit(train_data,train_target)
        modelFile = os.path.join(MODEL_PATH, 'keyword_model_'+w+'.pkl')
        OF = open(modelFile,'wb')
        pickle.dump(clf,OF,-1)
        print 'model saved to',modelFile
        print >>OF1, w+','+str(bestScore)+','+str(bestStd)
        if bestScore >= 0.75 and bestStd <= (bestScore-0.5)*0.6:
            print >>OF2,w+','+str(bestScore)+','+str(bestStd)
    OF1.close()
    OF2.close()
            
