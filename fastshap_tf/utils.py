from shap.utils._legacy import Link

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Lambda, Dropout, Multiply, BatchNormalization, Reshape, Concatenate, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import math
import numpy as np


class IdentityLinkTF(Link):
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x


class LogitLinkTF(Link):
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return tf.math.log(x/(1-x))

    @staticmethod
    def finv(x):
        return 1/(1+tf.math.exp(-x))


def convert_to_linkTF(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLinkTF()
    elif val == "logit":
        return LogitLinkTF()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"


class ShapleySampler(Layer):
    '''
    Layer to Sample S according to the Shapley Kernel Weights
    '''
    def __init__(self, num_features, paired_sampling=True, num_samples=1, **kwargs):
        super(ShapleySampler, self).__init__(**kwargs)
        
        self.num_features = num_features
        
        # Weighting kernel (probability of each subset size). 
        #credit = https://github.com/iancovert/sage/blob/master/sage/kernel_estimator.py
        w = tf.range(1, num_features)
        w = 1 / (w * (num_features - w))
        self.w = w / K.sum(w)
        
        self.paired_sampling = paired_sampling
        self.num_samples = num_samples
        
        self.ones_matrix = tf.linalg.band_part(
            tf.ones((num_features,num_features), tf.int32), 
            -1, 0)
    
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Sample subset size = number of features to select in each sample
        num_included = tf.random.categorical(
            tf.expand_dims(tf.math.log(self.w), 0), batch_size * self.num_samples
        )
        num_included = tf.transpose(num_included, [1,0])
        
        S = tf.gather_nd(self.ones_matrix, num_included)
        S = tf.map_fn(tf.random.shuffle, S)
        
        # Uniformly sample features of subset size
        S = tf.reshape(S, [batch_size, self.num_samples, self.num_features])
        
        #Paried Sampling 
        if self.paired_sampling:
            S_complement = 1 - S
            S = tf.concat([S, S_complement], axis = 1)
        
        return S
    
    def get_config(self):
        config = super(ShapleySampler, self).get_config()
        config.update({"num_features": self.num_features})
        config.update({"paired_sampling": self.paired_sampling})
        config.update({"num_samples": self.num_samples})
        config.update({"ones_matrix": self.ones_matrix})
        config.update({"w": self.w})
        return config     
    
    
class ResizeMask(Layer):
    def __init__(self, in_shape, mask_size, output_channels=1, **kwargs):
        super(ResizeMask, self).__init__(**kwargs)
        
        self.in_shape = in_shape
        self.mask_size = mask_size
        self.output_channels = output_channels
        self.reshape_shape, self.resize_aspect, self.pad_shape = self.get_reshape_shape()
        
    def get_reshape_shape(self):
        
        #Check if Multi Dimensional
        if type(self.in_shape) == int:
            out_shape = self.mask_size
            resize_aspect = int(math.ceil(self.in_shape/self.mask_size))
            
            #Get Pad Length Used
            resize_shape = out_shape * resize_aspect
            pad_shape = int((resize_shape - self.in_shape)/2)
            
            return out_shape, resize_aspect, pad_shape
        else:
            #Get Input Dimensions Ratio
            input_shape = np.array(list(self.in_shape)[:-1])
            gcd = np.gcd.reduce(input_shape)
            ratio = input_shape/gcd
            #Get Working Mask Size and Aspect Ratio
            mask_size = self.mask_size
            aspect = (mask_size/np.prod(ratio))**(1/len(ratio))
            out_shape = (ratio * aspect).astype(int)
            resize_aspect = int(math.ceil(gcd/aspect))
            
            #Get Pad Length Used
            resize_shape = out_shape * resize_aspect
            pad_shape = ((resize_shape - input_shape)/2).astype(int)
        
            return (*out_shape, self.output_channels), resize_aspect, pad_shape
    
    def call(self, inputs):
        
        if type(self.in_shape) == int:
            #Resize
            out = Lambda(
                lambda x: K.repeat_elements(x, rep = self.resize_aspect, axis = 1)
            )(inputs)
            
            #Slice to Input Size
            out = Lambda(lambda x: x[:, self.pad_shape:-self.pad_shape])(out)
            
        else:
            #Reshape
            out = Reshape(tuple(self.reshape_shape))(inputs)
            
            #Resize
            for i in range(len(self.reshape_shape)-1):
                out = Lambda(
                    lambda x: K.repeat_elements(x, rep = self.resize_aspect, axis = i+1)
                )(out)
                
            #Crop to Input Size
            if len(self.pad_shape) == 1:
                out = Lambda(lambda x: x[:, self.pad_shape[0]:-self.pad_shape[0], :])(out)
            elif len(self.pad_shape) == 2 and self.pad_shape[0] != 0:
                out = Lambda(
                    lambda x: x[:, 
                                self.pad_shape[0]:-self.pad_shape[0],
                                self.pad_shape[1]:-self.pad_shape[1],
                                :]
                )(out)
        
        
        return out
    
    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + list(self.in_shape)[:-1] + [self.output_channels])
    
    def get_config(self):
        config = super(ResizeMask, self).get_config()
        config.update({"in_shape": self.in_shape})
        config.update({"mask_size": self.mask_size})
        config.update({"output_channels": self.output_channels})
        config.update({"reshape_shape": self.reshape_shape})
        config.update({"resize_aspect": self.resize_aspect})
        config.update({"pad_shape": self.pad_shape})
        return config