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
from sklearn.svm import SVR,LinearSVR,NuSVR
import glob
import pandas as pd
import scipy.io as sio
import math
from sklearn.grid_search import GridSearchCV
import shutil
import matplotlib.pyplot as plt
import xgboost
from scipy import stats
from outliers import smirnov_grubbs as grubbs
from dataLoder import us10kLoader

from sklearn.decomposition import PCA

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from FaceFrontalisation.facefrontal import getDefaultFrontalizer


labelList = ['Old', 'Masculine', 'Baby-faced', 'Competent', 'Attractive', 'Energetic', \
             'Well-groomed', 'Intelligent', 'Honest', 'Generous', \
             'Trustworthy', 'Confident', 'Rich', 'Dominant']

FIG_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/Code/ML4RomeArt/Figures'
#MODEL_PREFIX = 'social_landmarks_XGB_MAPE_FRFULL'
#MODEL_PREFIX = 'social_landmarks_SVR_PWCA_FRNR'
#MODEL_PREFIX = 'social_landmarks_SVR_'
MODEL_PREFIX = 'social_landmarks_NUSVR_R2_FRFULL'
MODEL_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/models/iccv15'
DATASET_PREF = 'iccv15__'
TRAIN_DATA_MODE = 0

RESULT_PATH = 'keywordResults/'
ARG_LEN = 30

IMG_MODE = 1
REGRESSOR_MODE = 'SVR'
SCALE_MODE = 1
PLOT_D1 = 5
PLOT_D2 = 10

R2_THR = 0.0
FIG_SIZE = 100

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

def pair_wise_classification_accuracy(y_true, y_pred):
    ct = 0
    for i in xrange(y_true.shape[0]):
        for j in xrange(i+1,y_true.shape[0]):
            if (y_pred[i]-y_pred[j])*(y_true[i]-y_true[j]) > 0 or ( (y_true[i]-y_true[j]) == 0 and (y_pred[i]-y_pred[j]) ==0 ):
                ct+=1
    return float(ct)/((y_true.shape[0]-1)*y_true.shape[0]/2)

def quantized_classification_accuracy(y_true, y_pred, Q = 2):
    assert Q >= 2
    qt = [np.percentile(y_true, 100/Q*q) for q in xrange(1,Q)]
    ct = 0
    for i in xrange(y_true.shape[0]):
        lt = lp = -1
        for q,t in enumerate(qt):
            if y_true[i] < t:
                lt = q
                break
        for q,t in enumerate(qt):
            if y_pred[i] < t:
                lp = q
                break
        if lt<0:
            lt = len(qt)
        if lp<0:
            lp = len(qt)
        if lt == lp:
            ct += 1
    return float(ct)/y_true.shape[0]

def mape_xg(yhat, y):
    yt = y.get_label()
    return "median_absolute_percentage_error", median_absolute_percentage_error(yt, yhat)

def pwca_xg(yhat, y):
    yt = y.get_label()
    return "pair_wise_classification_error", 1-pair_wise_classification_accuracy(yt, yhat)

def customScorerMAPE(estimator, X,y):
    return -median_absolute_percentage_error(y,estimator.predict(X))

def customScorerPWCA(estimator, X,y):
    return pair_wise_classification_accuracy(y,estimator.predict(X))
def customScorerQCA2(estimator, X,y):
    return quantized_classification_accuracy(y,estimator.predict(X))
customScorer = customScorerMAPE
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

def prepareData(dataX, dataY, l,imNameSet):
    assert l >=0 and l < dataY.shape[1]
    if TRAIN_DATA_MODE == 0:
        dataYT = dataY[:,l]
        dataXT = dataX
    else:
        irow = 0
        dataXT,dataYT = [], []
        for i in xrange(dataX.shape[0]):
            if flist[i] in imNameSet:
                if not np.any(np.isnan(dataY[i,l])):
                    dataXT.append(dataX[irow,:])
                    dataYT.append(dataY[i,l])
                irow+=1
        dataXT = np.asarray(dataXT)
        dataYT = np.asarray(dataYT)
    return dataXT,dataYT

def fixAnno(anno):
    ave = np.nanmean(anno)
    assert not np.isinf(ave)
    for i in xrange(anno.shape[0]):
        if np.isnan(anno[i]):
            anno[i] = ave
    return anno


if __name__ == '__main__':
    assert len(sys.argv)>1
    if sys.argv[1] == 'train':
        assert len(sys.argv[2:]) >= 2,'usage: annoatation mat, pickled face feature'
        if TRAIN_DATA_MODE == 0:
            dataY = sio.loadmat(sys.argv[2])['trait_annotation']
        elif TRAIN_DATA_MODE == 1:
            labelList,flist,dataY = us10kLoader(sys.argv[2])
        else:
            assert False, 'unsupported TRAIN_DATA_MODE'
        imName,dataX = pickle.load(open(sys.argv[3],'rb'))
        imName,dataX = filterNaN(imName,dataX)
        imNameSet = set([os.path.split(imn)[1] for imn in imName])
        print np.max(dataX),np.min(dataX)
        if TRAIN_DATA_MODE == 0:
            assert dataX.shape[0] == dataY.shape[0]
        dataX = prep.minmax_scale(dataX)
        #baseReg = SVR(kernel='linear', gamma=0.1, coef0=0.0, C=1.0, epsilon=0.2, verbose=False, cache = 1000)
        #baseReg = LinearSVR(dual=True)
        baseReg = NuSVR(nu=0.5, C=1.0, kernel='rbf', cache_size=1000)
        numLabel = dataY.shape[1]
        wmscore = 0.0
        paramGridLinear = {'C':(0.01,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('linear',),'epsilon':(0.001,0.005,0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.8,1.0)}
        paramGridLinearC = {'C':(0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0,1000.0),'kernel':('linear',),'epsilon':(0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6)}
        paramGridLinearCC = {'C':(0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0),'kernel':('linear',),'epsilon':(0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6)}

        paramGridRbf = {'gamma':(0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1.0),'C':(0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.25,1.5,2.0,3.0,5.0,8.0,10.0,20.0,40.0,60.0,100.0,500.0,1000.0),'kernel':('rbf',),'epsilon':(0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.8,1.0)}
        paramGridRbfC = {'gamma':(0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0),'C':(0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0,1000.0),'kernel':('rbf',),'epsilon':(0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6)}
        paramGridRbfNuCC = {'gamma':(0.0001,0.0005,0.001,0.01,0.1,1.0,10.0),'C':(0.01,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0),'kernel':('rbf',),'nu':(0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99)}
        paramGridLinearNuCC = {'C':(0.01,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0),'kernel':('linear',),'nu':(0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99)}
        paramGridLibLinear = {'C':(0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0,1000.0),'epsilon':(0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6)}
        #paramGridAll = [paramGridLinear,paramGridRbf]
        #paramGridAll = [paramGridLinearC,paramGridRbfC]
        #paramGridAll = [paramGridLibLinear]
        paramGridAll = [paramGridRbfNuCC,paramGridLinearNuCC]
        for l in xrange(numLabel):
            print 'label',l,labelList[l]
            dataXT,dataYT = prepareData(dataX,dataY,l,imNameSet)
            print 'number of samples',dataXT.shape[0]
            print np.min(dataYT),np.max(dataYT),np.max(dataYT)-np.min(dataYT),np.std(dataYT)
            if REGRESSOR_MODE == 'XGB':
                trainX, valX, trainY, valY = cross.train_test_split(dataXT, dataYT, test_size=0.2, random_state=6174)
                gscv = xgboost.XGBRegressor(max_depth=3, learning_rate=0.005, n_estimators=3000, silent=True,
                                           objective='reg:linear', nthread=-1,subsample=0.6, colsample_bytree=0.6,seed=2333)
                eval_set=[(trainX, trainY), (valX, valY)]
                gscv.fit(trainX, trainY,eval_set = eval_set, eval_metric=mape_xg,early_stopping_rounds = 15,verbose=True)
                evals_result = gscv.evals_result()
                evals_result = [float(x) for x in evals_result['validation_1']['error']]
                print np.max(evals_result),np.argmax(evals_result)
                pred = gscv.predict(valX)
                print pair_wise_classification_accuracy(valY,pred)
                print median_absolute_percentage_error(valY,pred)
                print mean_absolute_percentage_error(valY,pred)
                print met.explained_variance_score(valY,pred)
                print met.r2_score(valY,pred)
                print quantized_classification_accuracy(valY,pred,2)
            else:
                gscv = GridSearchCV(baseReg,paramGridAll, scoring = 'r2',cv = 5, n_jobs=-1, refit = True,verbose = 1)
                gscv.fit(dataXT, dataYT)
                score = gscv.best_score_
                print gscv.best_params_
                print 'score for social attr',l,labelList[l],np.mean(score),'+-',np.std(score)
                wmscore += np.mean(score)
                
            savePath = os.path.join(MODEL_PATH,MODEL_PREFIX+str(l)+'_'+labelList[l]+'.pkl')
            OF = open(savePath,'wb')
            pickle.dump(gscv,OF,-1)
            OF.close()
            print 'model saved to',savePath
            sys.stdout.flush()
        print wmscore/numLabel
        
    elif sys.argv[1] == 'validation':
        assert len(sys.argv[2:]) >= 3,'usage: annoatation mat, pickled face feature, path to output the most and least face images'
        if TRAIN_DATA_MODE == 0:
            dataY = sio.loadmat(sys.argv[2])['trait_annotation']
        elif TRAIN_DATA_MODE == 1:
            labelList,flist,dataY = us10kLoader(sys.argv[2])
        else:
            assert False, 'unsupported TRAIN_DATA_MODE'
        imName,dataX = pickle.load(open(sys.argv[3],'rb'))
        imName,dataX = filterNaN(imName,dataX)
        imNameSet = set([os.path.split(imn)[1] for imn in imName])
        outputPath = sys.argv[4]
        dataX = prep.minmax_scale(dataX)
        if TRAIN_DATA_MODE == 0:
            assert dataX.shape[0] == dataY.shape[0]
        numLabel = dataY.shape[1]
        OF = open(os.path.join(RESULT_PATH,DATASET_PREF+MODEL_PREFIX+'_validation_results.csv'),'w')
        print >>OF, 'index,name,r2_score,pair_wise_classification_accuracy,quantized classification accuracy_binary'
        for l in xrange(numLabel):
            dataXT,dataYT = prepareData(dataX,dataY,l,imNameSet)
            savePath = os.path.join(MODEL_PATH,MODEL_PREFIX+str(l)+'_'+labelList[l]+'.pkl')
            rgr = pickle.load(open(savePath,'rb'))
            pred = rgr.predict(dataXT)
            argPredict = np.argsort(pred)
            argTrue = np.argsort(dataYT)
            print '>>>>>>>>>>>>>>>>>>>>'
            print 'attribute',labelList[l],'training metrics (E_in)'
            print 'mean,median,std for annotation',np.mean(dataYT),np.median(dataYT),np.std(dataYT)
            print 'mean,median,std for predicted',np.mean(pred),np.median(pred),np.std(pred)
            print 'explained_variance_score',met.explained_variance_score(dataYT,pred)
            print 'r2_score',met.r2_score(dataYT,pred)
            print 'mean_squared_error',met.mean_squared_error(dataYT,pred)
            print 'mean_absolute_error',met.mean_absolute_error(dataYT,pred)
            print 'median_absolute_error',met.median_absolute_error(dataYT,pred)
            print 'mean_absolute_percentage_error',mean_absolute_percentage_error(dataYT,pred)
            print 'median_absolute_percentage_error',median_absolute_percentage_error(dataYT,pred)
            print 'almost_correct_value_percentage',almost_correct_value_percentage(dataYT,pred,0.1)
            print 'almost_correct_rank_percentage',almost_correct_rank_percentage(dataYT,pred,10)
            print 'pair_wise_classification_accuracy', pair_wise_classification_accuracy(dataYT,pred)
            q = 2
            while q <= 32:
                print 'quantized classification accuracy',q,'classes',quantized_classification_accuracy(dataYT,pred,q)
                q *= 2
            ct = 0
            def copyFaceImg(num,idx,surf):
                opath = imName[idx]
                if not os.path.isfile(opath):
                    print opath,'is not a file'
                    return
                dpath = outputPath+'_'+surf+'_'+labelList[l]+'_'+str(num)+'.jpg'
                shutil.copy(opath,dpath)

            for n,i in enumerate(argPredict[:ARG_LEN]):
                copyFaceImg(n,i,'least')
                if i in argTrue[:ARG_LEN]:
                    ct+=1
            print 'least ones hit',ct
            ct = 0
            for n,i in enumerate(argPredict[-ARG_LEN:][::-1]):
                copyFaceImg(n,i,'most')
                if i in argTrue[-ARG_LEN:]:
                    ct+=1
            importantMetric = []
            print 'most ones hit',ct
            print '==================='
            print 'attribute',labelList[l],'cross validation metrics (E_val)'
            score = cross.cross_val_score(rgr.best_estimator_, dataXT, dataYT, scoring = 'r2',cv = 5, n_jobs=-1)
            print 'r2',np.mean(score),'+-',np.std(score)
            importantMetric.append(np.mean(score))
            score = cross.cross_val_score(rgr.best_estimator_, dataXT, dataYT, scoring = 'mean_squared_error',cv = 5, n_jobs=-1)
            print 'mean_squared_error',np.mean(score),'+-',np.std(score)
            score = cross.cross_val_score(rgr.best_estimator_, dataXT, dataYT, scoring = 'median_absolute_error',cv = 5, n_jobs=-1)
            print 'median_absolute_error',np.mean(score),'+-',np.std(score)
            score = cross.cross_val_score(rgr.best_estimator_, dataXT, dataYT, scoring = customScorerPWCA,cv = 5, n_jobs=-1)
            print 'pair_wise_classification_accuracy',np.mean(score),'+-',np.std(score)
            importantMetric.append(np.mean(score))
            score = cross.cross_val_score(rgr.best_estimator_, dataXT, dataYT, scoring = customScorerQCA2,cv = 5, n_jobs=-1)
            print 'quantized classification accuracy - binary',np.mean(score),'+-',np.std(score)
            importantMetric.append(np.mean(score))

            print >>OF, str(l)+','+labelList[l]+','+','.join([str(m) for m in importantMetric])
        OF.close()
    elif sys.argv[1] == 'transfer':
        if SCALE_MODE == 0:
            assert len(sys.argv[2:]) >= 2, 'usage: pickled statue face feature, path to write the social evaluation result'
            pp = 2
        else:
            assert len(sys.argv[2:]) >= 3, 'usage: pickled statue face feature, path to write the social evaluation result, pickled photo face feature'
            pp = 3
        if TRAIN_DATA_MODE != 0:
            assert len(sys.argv[2:]) >= pp+1, 'must provide label list txt file'
            labelList,flist,dataY = us10kLoader(sys.argv[-1])
        imName,dataX = pickle.load(open(sys.argv[2],'rb'))
        imName,dataX = filterNaN(imName,dataX)
        if SCALE_MODE == 1:
            imNameP,dataXP = pickle.load(open(sys.argv[4],'rb'))
            scaler = prep.MinMaxScaler()
            scaler.fit(dataXP)
            dataX = scaler.transform(dataX)
        else:
            dataX = prep.minmax_scale(dataX)
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

    elif sys.argv[1] == 'show':
        assert len(sys.argv[2:]) >= 3,'usage: pickled social evaluation results, outputpath, good list path'
        imName,socialEval = pickle.load(open(sys.argv[2],'rb'))
        outputPath = sys.argv[3]
        goodPath = sys.argv[4]
        goodList = set([s.strip() for s in open(goodPath,'r').readlines()])
        if TRAIN_DATA_MODE != 0:
            assert len(sys.argv[2:]) >= 4,'must provide label list txt file'
            labelList,flist,dataY = us10kLoader(sys.argv[-1])
        attrRange = np.max(socialEval),np.min(socialEval)
        plt.figure(figsize=(10,20))
        for i,n in enumerate(imName):
            s = os.path.split(n)[1]
            if s not in goodList:
                continue
            y_pos = np.arange(len(labelList))
            plt.clf()
            plt.barh(y_pos, socialEval[i,:], facecolor='#9999ff', edgecolor='white')
            plt.yticks(y_pos, labelList)
            plt.xlabel('predicted values')
            for y in y_pos:
                plt.text(socialEval[i,y] + 0.1, y + 0.6, '%.2f' % socialEval[i,y], ha='center', va='bottom')
            plt.xlim(attrRange[1]-0.1, attrRange[0]+0.1)
            plt.savefig(os.path.join(outputPath,s+'_'+DATASET_PREF+'_Attributes.png'))

    elif sys.argv[1] == 'pca':
        assert len(sys.argv[2:]) >= 3,'usage: pickled keyword analysis result, pickled social evalution result, validation result'
        if TRAIN_DATA_MODE != 0:
            assert len(sys.argv[2:]) >= 4,'must provide label list txt file'
            labelList,flist,dataY = us10kLoader(sys.argv[-1])
        numLabel = len(labelList)
        portId,stats, wdCount = pickle.load(open(sys.argv[2],'rb'))
        imName,socialEvalOri = pickle.load(open(sys.argv[3],'rb'))
        val = open(sys.argv[4],'r').readlines()
        okCols = []
        for i,line in enumerate(val[1:]):
            sp = line.split(',')
            r2 = float(sp[2])
            if r2 >= R2_THR:
                okCols.append(i)
            else:
                print 'drop column:',sp[0],sp[1]
        socialEval = np.ndarray((socialEvalOri.shape[0],len(okCols)), dtype = 'float')
        for i,c in enumerate(okCols):
            socialEval[:,i] = socialEvalOri[:,c]
        numLabel = len(okCols)
        print 'new column number:',numLabel
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
        portPCA = PCA(2, whiten = False)
        eval2D = portPCA.fit_transform(commonEval)
        print 'feature mean',portPCA.mean_,'ev ratio',portPCA.explained_variance_ratio_
        coEval = np.concatenate([commonEval,eval2D], axis = 1)
        corr = np.corrcoef(coEval,rowvar = 0)
        evalCoord = corr[:numLabel, -2:]
        kwCoord = np.ndarray((commonWC.shape[1],2), dtype = 'float')
        print 'number of keywords', commonWC.shape[1]
        for i in xrange(commonWC.shape[1]):
            haveEval = commonEval[commonWC[:,i] > 0,:]
            meanEval = (np.nanmean(haveEval, 0)-portPCA.mean_).reshape(1,numLabel)
            meanEval = prep.normalize(meanEval, norm = 'l1', axis = 1)
            kwCoord[i,:] = np.dot(meanEval, evalCoord)
        plt.figure(figsize=(FIG_SIZE,FIG_SIZE))
        plt.scatter(evalCoord[:,0],evalCoord[:,1],s=300,label='traits', marker = 'o', color = 'r')
        for i in xrange(numLabel):
            plt.text(evalCoord[i,0],evalCoord[i,1],labelList[okCols[i]] , ha='center', va='bottom')
        plt.scatter(kwCoord[:,0],kwCoord[:,1], s=50, label = 'keywords', marker = 'x', color = 'b')
        for i in xrange(commonWC.shape[1]):
            plt.text(kwCoord[i,0],kwCoord[i,1],stats[i][0], fontsize = 'xx-small',ha='center', va='bottom')
        plt.grid(True, linestyle = 'solid', linewidth = FIG_SIZE/10)
        plt.legend()
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        #plt.show()
        plt.savefig(os.path.join(FIG_PATH,DATASET_PREF+'_Attributes_Keywords.png'))
    elif sys.argv[1] == 'analysis':
        assert len(sys.argv[2:]) >= 2,'usage: pickled keyword analysis result, pickled social evalution result'
        if TRAIN_DATA_MODE != 0:
            assert len(sys.argv[2:]) >= 3,'must provide label list txt file'
            labelList,flist,dataY = us10kLoader(sys.argv[-1])
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
        OF = open(RESULT_PATH+DATASET_PREF+'correlationKeywordsSocialEval.csv','w')
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

        OF = open(RESULT_PATH+DATASET_PREF+'keywordSocialCorrPairsSorted.csv','w')
        for k,s,c in corrPairs:
            tstr = k+','+s+','+ "{:.4f}".format(c)
            print>>OF,tstr
        OF.close()
    elif sys.argv[1] == 'argsort':
        assert len(sys.argv[2:]) >= 3,'usage: pickled social evaluation results, outputpath, croped face path'
        imName,socialEval = pickle.load(open(sys.argv[2],'rb'))
        outputPath = sys.argv[3]
        facePath = sys.argv[4]
        if TRAIN_DATA_MODE != 0:
            assert len(sys.argv[2:]) >= 4,'must provide label list txt file'
            labelList,flist,dataY = us10kLoader(sys.argv[-1])
            
        def copyFaceImg(idx,surf,lab, asort):
            fname = os.path.split(imName[asort[idx]])[1]
            if IMG_MODE == 1:
                opath = os.path.join(facePath,fname+'_frontal.jpg')
            else:
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
                copyFaceImg(n,'least',l, asort)
                copyFaceImg(-(n+1),'most',l, asort)
    elif sys.argv[1] == 'compare':
        assert len(sys.argv[2:]) >= 3,'usage: pickled social evaluation results for statues, pickled feature for photos, photo annotation'
        imName,socialEval = pickle.load(open(sys.argv[2],'rb'))
        imName,dataX = pickle.load(open(sys.argv[3],'rb'))
        dataX = prep.minmax_scale(dataX)
        if TRAIN_DATA_MODE == 0:
            dataY = sio.loadmat(sys.argv[4])['trait_annotation']
        elif TRAIN_DATA_MODE == 1:
            labelList,flist,dataY = us10kLoader(sys.argv[4])
            lines = open(os.path.join(RESULT_PATH,DATASET_PREF+MODEL_PREFIX+'_validation_results.csv'),'r').readlines()[1:]
        else:
            assert False, 'unsupported TRAIN_DATA_MODE'
        fig = plt.figure(1, figsize=(36, 18))
        numLabel = len(labelList)
        for l in xrange(numLabel):
            savePath = os.path.join(MODEL_PATH,MODEL_PREFIX+str(l)+'_'+labelList[l]+'.pkl')
            rgr = pickle.load(open(savePath,'rb'))
            print ">>>>>>>>>>"
            print 'statistics for',labelList[l]
            if TRAIN_DATA_MODE == 1:
                print 'validation r2_score is',lines[l].split(',')[2]
            def printStat(arr,name):
                print name
                tmean,tstd = np.mean(arr),np.std(arr)
                print 'mean',tmean,'median',np.median(arr),'std',tstd
            def normalCheck(arr, name):
                print name
                p1 = stats.shapiro(arr)[1]
                p2 = stats.normaltest(arr)[1]
                if p1+p2 > 0.9:
                    res = 'normally distributed'
                else:
                    res = 'not normal'
                print 'p-value from W-test and normal-test',p1,p2,res
            def ZCheck(a,b):
                e1,e2 = np.mean(a),np.mean(b)
                s1,s2 = np.std(a),np.std(b)
                n1,n2 = a.shape[0],b.shape[0]
                Z = (e1-e2)/np.sqrt(s1/n1+s2/n2)
                return np.abs(Z)
                
            pred = rgr.predict(dataX)
            anno = fixAnno(dataY[:,l])
            printStat(anno,'annotation')
            printStat(pred,'predicted')
            trans = socialEval[:,l]
            #print trans.shape
            #trans = grubbs.test(trans, 0.1)
            #print trans.shape
            printStat(trans,'transfered')

            print "**********"
            print "normal Test for",labelList[l]
            normalCheck(anno,'annotation')
            normalCheck(pred,'predicted')
            normalCheck(trans,'transfered')

            Z = ZCheck(pred,trans)
            print "-----------"
            print "Z-Test for",labelList[l]
            print '|Z| is',Z
            if Z>=2.58:
                print 'very significant difference'
            elif Z >= 1.96:
                print 'significant difference'
            else:
                print 'no significant difference'
            print 'creating boxplot for',labelList[l]
            data_to_plot = [anno, pred, trans]
            # Create an axes instance
            ax = fig.add_subplot(PLOT_D1,PLOT_D2,l+1)
            ax.set_title(labelList[l])
            ## add patch_artist=True option to ax.boxplot() 
            ## to get fill color
            bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans = True)

            ## change outline color, fill color and linewidth of the boxes
            for box in bp['boxes']:
                # change outline color
                box.set( color='#7570b3', linewidth=2)
                # change fill color
                box.set( facecolor = '#1b9e77' )
                ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
                whisker.set(color='y', linewidth=2)

            ## change color and linewidth of the caps
            for cap in bp['caps']: #caps
                cap.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the medians
            for median in bp['medians']: #medians
                median.set(color='r', linewidth=2)

                ## change the style of fliers and their fill
            for flier in bp['fliers']: #fliers
                flier.set(marker='o', color='k', alpha=0.5)
        fig.savefig(os.path.join(FIG_PATH,DATASET_PREF+'_'+os.path.split(sys.argv[2])[1]+'_compare.png'))
        plt.show(fig)
    else:
        print 'please use train, transfer, validation, analysis, argsort or compare command'
    
