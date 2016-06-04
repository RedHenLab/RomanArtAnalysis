# -*- coding: utf-8 -*-
#Forked from https://github.com/tdeboissiere/Kaggle/blob/master/StateFarm/DeconvNet/

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import KerasDeconv
import cPickle as pickle
from utils import get_deconv_images
from utils import plot_deconv
from utils import plot_max_activation
from utils import find_top9_mean_act
import glob
import cv2
import sys,os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
from CNN4Portraits_keras import get_im,cache_data,restore_data,read_model_any,mean_pixel,preprocess_data
from sampleSubsetImages import sampleEqualSubset
# color type: 1 - grey, 3 - rgb
color_type_global = 1

global_rows_cols = (64 , 64)
DRAW_NUM = 256

if __name__ == "__main__":

    ######################
    # Misc
    ######################
    model = None  # Initialise VGG model to None
    Dec = None  # Initialise DeconvNet model to None
    if not os.path.exists("./Figures/"):
        os.makedirs("./Figures/")
    ############
    # Load data
    ############
    if len(sys.argv) == 5:
        list_img = sampleEqualSubset(sys.argv[1],DRAW_NUM/2, int(sys.argv[4]))
    else:
        list_img = glob.glob(os.path.join(sys.argv[1],"*.jpg"))
    assert len(list_img) > 0, "no images in the folder"
    data = []
    for im_name in list_img:
        im = get_im(im_name,global_rows_cols[0],global_rows_cols[1],color_type_global)
        data.append(im)
    data = np.array(data,dtype= 'float')
    data = preprocess_data(data,global_rows_cols[0],global_rows_cols[1],color_type_global)
    
    archF = sys.argv[2]
    weightsF = sys.argv[3]

    if not model:
        model = read_model_any(archF,weightsF)
    if not Dec:
        Dec = KerasDeconv.DeconvNet(model)

    print 'haha'

    ###############################################
    # Action 1) Get max activation for a slection of feat maps
    ###############################################
    get_max_act = False
    if get_max_act:
        if not model:
            model = load_model('./Data/vgg16_weights.h5')
        if not Dec:
            Dec = KerasDeconv.DeconvNet(model)
        d_act_path = './Data/dict_top9_mean_act.pickle'
        d_act = {"convolution2d_13": {},
                 "convolution2d_10": {}
                 }
        for feat_map in range(10):
            d_act["convolution2d_13"][feat_map] = find_top9_mean_act(
                data, Dec, "convolution2d_13", feat_map, batch_size=32)
            d_act["convolution2d_10"][feat_map] = find_top9_mean_act(
                data, Dec, "convolution2d_10", feat_map, batch_size=32)
            with open(d_act_path, 'w') as f:
                pickle.dump(d_act, f)

    ###############################################
    # Action 2) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    deconv_img = False
    if deconv_img:
        d_act_path = './Data/dict_top9_mean_act.pickle'
        d_deconv_path = './Data/dict_top9_deconv.pickle'
        if not model:
            model = load_model('./Data/vgg16_weights.h5')
        if not Dec:
            Dec = KerasDeconv.DeconvNet(model)
        get_deconv_images(d_act_path, d_deconv_path, data, Dec)

    ###############################################
    # Action 3) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    plot_deconv_img = False
    if plot_deconv_img:
        d_act_path = './Data/dict_top9_mean_act.pickle'
        d_deconv_path = './Data/dict_top9_deconv.npz'
        target_layer = "convolution2d_10"
        plot_max_activation(d_act_path, d_deconv_path, data, target_layer, save=True)

    ###############################################
    # Action 4) Get deconv images of some images for some
    # feat map
    ###############################################
    deconv_specific = True
    if deconv_specific:
        target_layer = "dense_1"
        feat_map = 0
        num_img = DRAW_NUM
        np.random.seed()
        img_index = np.random.choice(data.shape[0], num_img, replace=False)
        img_index = sorted(img_index)
        plot_deconv(img_index, data, Dec, target_layer, feat_map, save = True)
        plot_deconv(img_index, data, Dec, target_layer, 1-feat_map, save = True)

