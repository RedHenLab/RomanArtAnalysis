# -*- coding: utf-8 -*-

import numpy as np

import sys,os
import glob
import cv2
import math
import cPickle as pickle
import datetime
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss,accuracy_score
from numpy.random import permutation
import keras

np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 1
global_split_random = 2016
# color_type = 1 - gray
# color_type = 3 - RGB

global_rows_cols = (64 , 64)

mean_pixel = [103.939, 116.779, 123.68]
DATASET_PATH = '/home/mfs6174/GSOC2016/GSoC2016-RedHen/dataset'


def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    # resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized


def get_po_data(dataStr):
    dr = dict()
    path = os.path.join(DATASET_PATH,dataStr+'.txt')
    print('Read dataset index')
    lines = open(path, 'r').readlines()
    return [(line.strip().split()[0],int(line.strip().split()[1])) for line in lines]


def load_train(img_rows, img_cols, dataStr, color_type=1):
    X_train = []
    y_train = []

    po_data = get_po_data(dataStr)

    print('Read train images')
    for (fl,j) in po_data:
        img = get_im(fl, img_rows, img_cols, color_type)
        X_train.append(img)
        y_train.append(j)

    return X_train, y_train




def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model, index, cross, dataStr):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = dataStr+'_architecture' + str(index) + cross + '.json'
    weight_name = dataStr+ '_model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model_any(arch, weights):
    model = model_from_json(open(os.path.join('cache', arch)).read())
    model.load_weights(os.path.join('cache', weights))
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def split_validation_set(train, target, test_size):
    random_state = global_split_random
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test


def read_and_normalize_and_shuffle_train_data(img_rows, img_cols, dataStr,
                                              color_type=1,shuffle = True):

    cache_path = os.path.join('cache', 'train_dataName_'+dataStr+'_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target = \
            load_train(img_rows, img_cols, dataStr, color_type)
        cache_data((train_data, train_target),
                   cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target) = \
            restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)


    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    numLabel = int(dataStr[dataStr.rfind('-')+1:])
    train_target = np_utils.to_categorical(train_target, numLabel)
    train_data = train_data.astype('float32')
    if color_type != 1:
        for c in range(3):
            train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    else:
        train_data = train_data - np.mean(mean_pixel)
    train_data /= 255.0
    if shuffle:
        perm = permutation(len(train_target))
        train_data = train_data[perm]
        train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, numLabel




def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('../pre-trained/vgg16_weights.h5')

    # Code above loads pre-trained data and
    def popLayer(model):
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
    for i in xrange(1):
        popLayer(model)
    model.add(Dense(2, activation='softmax'))
    #print model.summary() 
    # Learning rate is changed to 0.001
    sgd = SGD(lr=0.25e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg_simple_model(img_rows, img_cols, color_type=1, outputLen = 2, _lr = 1e-2):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(32, 3, 3, activation='relu', W_regularizer=l2(0.005)))
    #model.add(PReLU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(0.005)))
    #model.add(PReLU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
        
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', W_regularizer=l2(0.005)))
    #model.add(PReLU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l2(0.005)))
    #model.add(PReLU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(outputLen, activation='softmax', W_regularizer=l2(0.01)))

    #model.load_weights('../pre-trained/vgg16_weights.h5')

    # Code above loads pre-trained data and
    print model.summary() 
    # Learning rate is changed to 0.001
    sgd = SGD(lr=_lr, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def forceBalance(x,y):
    print 'force the dataset to be class balanced'
    ml = y.shape[1]-1
    print 'top label',ml
    count = np.array([np.nonzero(y[:,i])[0].shape[0] for i in xrange(ml+1)])
    large = np.argmax(count)
    print 'label count:',count
    print 'large label',large
    multi = [0 for _ in xrange(count.shape[0])]
    for i in xrange(count.shape[0]):
        if i != large:
            bm = np.floor(float(count[large])/count[i])
            tm = np.ceil(float(count[large])/count[i])
            if abs(bm*count[i]-count[large]) < abs(tm*count[i]-count[large]):
                multi[i] = int(bm)
            else:
                multi[i] = int(tm)
    new_data, new_target = [], []
    for idx in xrange(y.shape[0]):
        new_data.append(x[idx,:,:,:])
        new_target.append(y[idx,:])
        tt = np.nonzero(y[idx,:])[0][0]
        if multi[tt] > 0:
            new_data+=[x[idx,:,:,:] for _ in xrange(multi[tt]-1)]
            new_target+=[y[idx,:] for _ in xrange(multi[tt]-1)]
    x = np.asarray(new_data, dtype = np.float32)
    y = np.asarray(new_target,dtype = np.uint8)
    print 'new x shape',x.shape
    print 'new y shape',y.shape
    return x,y

class simplePrint(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print 'epoch',epoch,'done, loss',logs['loss'],'acc',logs['acc'],'val_loss',logs['val_loss'],'val_acc',logs['val_acc']
        
def run_cross_validation(dataStr, nfolds=5, nb_epoch=10, modelStr='', lr = 1e-2):

    # input image dimensions
    #img_rows, img_cols = 224, 224
    img_rows, img_cols = global_rows_cols
    batch_size = 128
    random_state = global_split_random

    train_data, train_target, outputLen = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols, dataStr,
                                                  color_type_global, shuffle = False)

    num_fold = 0
    kf = KFold(train_data.shape[0], n_folds=nfolds,
               shuffle=True, random_state=random_state)
    for train_po, test_po in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        #train_data_d, train_target_d = train_data[train_po], train_target[train_po]
        #val_data, val_target = train_data[test_po], train_target[test_po]
        train_data_d, train_target_d = forceBalance(train_data[train_po], train_target[train_po])
        val_data, val_target = forceBalance(train_data[test_po], train_target[test_po])
        model = vgg_simple_model(img_rows, img_cols, color_type_global, outputLen, lr)
        print 'model loaded'
        checkPoints = ModelCheckpoint('cache/'+dataStr+'_'+modelStr+'_fold_'+str(num_fold)+'weights.best.hdf5',
                                      monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlyStops = EarlyStopping(monitor='val_acc', patience=150, verbose=1, mode='max')
        simplePrints = simplePrint()
        model.fit(train_data_d, train_target_d, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=0,
                  callbacks = [checkPoints,earlyStops,simplePrints],
                  validation_split=0.0, shuffle=True, validation_data = (val_data,val_target))


        save_model(model, num_fold, modelStr, dataStr)




def cv_validate_with_best_epoch(dataStr, best_models, save_path = None):
    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = global_rows_cols
    batch_size = 256
    random_state = global_split_random
    nfolds = len(best_models)
    train_data, train_target, outputLen = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,dataStr,
                                                  color_type_global,shuffle = False)
    num_fold = 0
    kf = KFold(train_data.shape[0], n_folds=nfolds,
               shuffle=True, random_state=random_state)
    predictions_all = np.ndarray((train_data.shape[0], outputLen), dtype=np.float32) 
    for train_po, test_po in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        val_data, val_target = train_data[test_po], train_target[test_po]
        model = read_model_any(best_models[num_fold-1][0],best_models[num_fold-1][1])
        print 'model loaded'        
        predictions_valid = model.predict(val_data, batch_size=batch_size, verbose=1)
        predictions_all[test_po]=predictions_valid
    if save_path is not None:
        cache_data((predictions_all,np.argmax(train_target,axis = 1)), save_path)
    score = log_loss(train_target, predictions_all)
    accuracy = accuracy_score(np.argmax(train_target,axis = 1), np.argmax(predictions_all,axis = 1))
    print('Score log_loss, accuracy: ', score,accuracy)
    return predictions_all


# nfolds, nb_epoch, split
run_cross_validation('merge.gender-2',10, 500, 'vgg_5_0.5e-2_10x500',lr = 0.5e-2)


def cv_validate_with_best_epoch_wrapper(dataStr, modelStr, nFold, cache_path):
    models = [(dataStr+'_architecture'+str(i+1)+modelStr+'.json', dataStr+'_'+modelStr+'_fold_'+str(i+1)+'weights.best.hdf5') for i in xrange(nFold)]
    cv_validate_with_best_epoch(dataStr, models, cache_path)

cache_path = os.path.join('cache', 'val_merge.gender-2_bestval.dat')
cv_validate_with_best_epoch_wrapper('merge.gender-2','vgg_5_0.5e-2_10x500',10,cache_path)
#pred = cv_validate_with_best_epoch(models, cache_path)
#test_model_and_submit_any_wrapper_he('_vgg_16_full_3x12_0.025val', 2, 'high_epoch_vgg_16_full_0.25e-3_2x12_0.025val')
#test_model_and_submit_any_wrapper_he('_vgg_16_full_3x12_0.025val', 3, 'high_epoch_vgg_16_full_0.25e-3_3x12_0.025val')
