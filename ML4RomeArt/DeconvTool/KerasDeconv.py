# -*- coding: utf-8 -*-
#Forked from https://github.com/tdeboissiere/Kaggle/blob/master/StateFarm/DeconvNet/
import keras.backend as K
import time
import numpy as np
np.set_printoptions(precision=2)

SOFT_MAX_EPS = 1e-6

class DeconvNet(object):
    """DeconvNet class"""

    def __init__(self, model):
        self.model = model
        list_layers = self.model.layers
        self.lnames = [l.name for l in list_layers]
        assert len(self.lnames) == len(set(self.lnames)), "Non unique layer names"
        # Dict of layers indexed by layer name
        self.d_layers = {}
        for l_name, l in zip(self.lnames, list_layers):
            self.d_layers[l_name] = l

        # Tensor for function definitions
        self.x = K.T.tensor4('x')
        self.mask = K.T.tensor4('mask')
        self.flatX = K.T.matrix('flatX')

    def __getitem__(self, layer_name):
        try:
            return self.d_layers[layer_name]
        except KeyError:
            print "Erroneous layer name"

    def _deconv(self, X, lname, conv_input_shape = None, W = None):
        if conv_input_shape is None:
            o_width, o_height = self[lname].output_shape[-2:]
        else:
            o_width, o_height = conv_input_shape[-2:]
        if W is None:
            # Get filters. No bias for now
            W = self[lname].W
        # Get filter size
        if conv_input_shape is None:
            f_width = self[lname].W_shape[2]
            f_height = self[lname].W_shape[3]
        else:
            f_width = f_height = conv_input_shape[2]

        # Compute padding needed
        i_width, i_height = X.shape[-2:]
        pad_width = (o_width - i_width + f_width - 1) / 2
        pad_height = (o_height - i_height + f_height - 1) / 2

        assert isinstance(pad_width, int), "Pad width size issue at layer %s" % lname
        assert isinstance(pad_height, int), "Pad height size issue at layer %s" % lname
        if self.lnames.index(lname) != len(self.lnames)-1: 
            # Get activation function, do not apply to last layer (softmax)
            activation = self[lname].activation
            X = activation(X)
        # Transpose filter
        W = W.transpose([1, 0, 2, 3])
        W = W[:,:, ::-1, ::-1]
        # CUDNN for conv2d ?
        conv_out = K.T.nnet.conv2d(input=self.x, filters=W, border_mode='valid')
        # Add padding to get correct size
        pad = K.function([self.x,K.learning_phase()], K.spatial_2d_padding(
            self.x, padding=(pad_width, pad_height), dim_ordering="th"))
        X_pad = pad([X,1])
        # Get Deconv output
        deconv_func = K.function([self.x,K.learning_phase()], conv_out)
        X_deconv = deconv_func([X_pad,1])
        assert X_deconv.shape[-2:] == (o_width, o_height),\
            "Deconv output at %s has wrong size" % lname
        return X_deconv

    def _dedense(self, X, lname, input_c = None):
        fan_in = np.prod(self[lname].input_shape[1:])
        if input_c is None:
            input_c = fan_in
        width = height = int(np.sqrt(fan_in/input_c))
        assert abs(width-np.sqrt(fan_in/input_c))==0, \
            'feature map must be square to deal with dense layer'
        conv_input_shape = (None, input_c, width,height)
        fan_out = self[lname].output_shape[1]
        X = X.reshape((X.shape[0],fan_out,1,1))
        # Get filters. No bias for now
        W = K.reshape(self[lname].W,(fan_out, input_c, width, height))
        ret = self._deconv(X,lname,conv_input_shape,W)
        return ret.reshape((ret.shape[0],-1))
        

    def _forward_pass(self, X, target_layer):

        # For all layers up to the target layer
        # Store the max activation in switch
        d_switch = {}
        layer_index = self.lnames.index(target_layer)
        for lname in self.lnames[:layer_index + 1]:
            # Get layer output
            layer = self[lname]
            inc, out = layer.input, layer.output
            f = K.function([inc,K.learning_phase()], out)
            Y = f([X,1])
            if "maxpooling2d" in lname:
                #print lname,layer.strides,layer.pool_size
                #print X.shape,Y.shape
                strides, pool_size = layer.strides,layer.pool_size
                assert strides[0] >= pool_size[0] and strides[1] >= pool_size[1],\
                    "strides must be greater than or equal to pool_size, layer %s" % lname
                d_switch[lname] = np.zeros(X.shape,dtype = 'int')
                for didx in xrange(X.shape[0]):
                    for blocks_x in xrange(Y.shape[2] ):
                        for blocks_y in xrange( Y.shape[3] ):
                            this_block = X[didx,:, ( blocks_x  ) * strides[0] : blocks_x * strides[0]+pool_size[0],
                                           ( blocks_y  ) * strides[1] : blocks_y * strides[1]+pool_size[1] ]
                            amax = np.argmax(this_block.reshape((this_block.shape[0],-1)),axis = 1)
                            for ch in xrange(X.shape[1]):
                                d_switch[lname][didx,ch,( blocks_x  ) * strides[0] + amax[ch]/pool_size[1],( blocks_y  ) * strides[1] + amax[ch]%pool_size[1]] = 1
            X = Y
        return d_switch

    def _backward_pass(self, X, target_layer, d_switch, feat_map):
        # Run deconv/maxunpooling until input pixel space
        layer_index = self.lnames.index(target_layer)
        # Get the output of the target_layer of interest
        if layer_index == len(self.lnames)-1:
            print 'use none softmax'
            layer_output = K.function([self[self.lnames[0]].input,K.learning_phase()], K.dot(self[self.lnames[-2]].output,self[target_layer].W)+self[target_layer].b)
        else:
            layer_output = K.function([self[self.lnames[0]].input,K.learning_phase()], self[target_layer].output)
        X_outl = layer_output([X,1])
        # Special case for the starting layer where we may want
        # to switchoff somes maps/ activations
        softmax_offset = np.log((1.0/(1.0-SOFT_MAX_EPS)-1.0)/(X_outl.shape[1]-1))
        print "Deconvolving from %s..." % target_layer
        if feat_map is not None:
            print "Set other activation than channel %d to 0" % feat_map
            for i in range(X_outl.shape[1]):
                if i != feat_map:
                    if len(X_outl.shape)<4:
                        X_outl[:,i] = 0
                    else:
                        X_outl[:,i,:,:] = 0
                else:
                    if layer_index == len(self.lnames)-1:
                        print 'use softmax offset',softmax_offset
                        X_outl[:,i] = abs(softmax_offset)
                    
        #print X_outl[:,feat_map]
        #print X_outl[:,1-feat_map]
        # Iterate over layers (deepest to shallowest)
        batch_size = X_outl.shape[0]
        for lname in self.lnames[:layer_index+1][::-1]:
            print "Deconvolving %s..." % lname
            # Unpool, Deconv or do nothing; unflatten, see dense as conv
            if "flatten" in lname:
                ishape = self[lname].input_shape
                unflatten = K.function([self.flatX,K.learning_phase()], K.reshape(self.flatX, [batch_size]+[s for s in ishape[1:]]))
                X_outl = unflatten([X_outl,1])
            elif "maxpooling2d" in lname:
                p1, p2 = self[lname].pool_size
                uppool = K.function([self.x,self.mask,K.learning_phase()], K.resize_images(self.x, p1, p2, "th")*self.mask)
                X_outl = uppool([X_outl,d_switch[lname],1])
                #print d_switch[lname][0,feat_map,:,:]
                #print X_outl[0,feat_map,:,:]
            elif "convolution2d" in lname:
                X_outl = self._deconv(X_outl, lname)
            elif "padding" in lname:
                pass
            elif "dropout" in lname:
                pass
            elif "dense" in lname:
                X_outl = self._dedense(X_outl, lname, input_c = 256)
            else:
                raise ValueError(
                    "Invalid layer name: %s \n Can only handle maxpool and conv" % lname)
            print X_outl.shape
        return X_outl

    def get_layers(self):
        list_layers = self.model.layers
        list_layers_name = [l.name for l in list_layers]
        return list_layers_name

    def get_deconv(self, X, target_layer, feat_map=None):

        # First make predictions to get feature maps
        self.model.predict(X)
        # Forward pass storing switches
        print "Starting forward pass..."
        start_time = time.time()
        d_switch = self._forward_pass(X, target_layer)
        end_time = time.time()
        print 'Forward pass completed in %ds' % (end_time - start_time)
        # Then deconvolve starting from target layer
        X_out = self._backward_pass(X, target_layer, d_switch, feat_map)
        return X_out

