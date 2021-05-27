import numpy as np
import math

from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Lambda, Dropout, Multiply, BatchNormalization, Reshape, Concatenate, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from datetime import datetime
import os

from utils import convert_to_linkTF

from tensorflow.keras.applications.resnet50 import ResNet50


##########################################################################

class Surrogate:
    
    '''
    Wrapper around surrogate model.
    Args:
      surrogate:
    '''
    
    def __init__(self, 
                 original_model,
                 value_model, 
                 num_features, 
                 model_dir = None):
        
        # Models
        self.original_model = original_model
        self.value_model = value_model
        self.P = num_features
        
        
        # model save dir
        if model_dir is None:
            self.save = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.model_dir = os.path.join(os.getcwd(), self.save)
        else:
            self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
              
        
    def train(self, 
              train_data, 
              val_data, 
              max_epochs, 
              batch_size, 
              lookback,
              lr = 1e-3):
    
        # Training Parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lookback = lookback
        
        
        # Labels
        fx_train = self.original_model.predict(train_data)
        fx_val = self.original_model.predict(val_data)
        
        #################################################################
        
        #Make Model w/ Masking
        self.value_model.trainable = True
        
        model_input = Input(shape=self.P, dtype='float32', name='input')
        S = ShapleySampler(self.P, paired_sampling=False, num_samples=1)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32))(S)
        S = Reshape((self.P,))(S)
        xs = Multiply()([model_input, S])
        
        out = self.value_model(xs)
        
        self.model = Model(model_input, out)
        
        # Metrics
        METRICS = [ 
          tf.keras.metrics.AUC(name='auroc'),
          tf.keras.metrics.AUC(curve='PR', name='auprc'),
          tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
        ]
        
        # Model Checkpointing
        weights_path = os.path.join(self.model_dir, 'value_weights.h5')
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, 
                                     verbose=1, mode='min', cooldown=1, min_lr=self.lr/100)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=self.lookback) 
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(self.lr)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=OPTIMIZER,
            metrics=METRICS,
        )
        
        # Train Model
        self.model.fit(x = train_data,
                       y = fx_train,
                       epochs = self.max_epochs,
                       batch_size = self.batch_size,
                       validation_data = (val_data, fx_val),
                       callbacks = CALLBACKS)
        
        # Get Checkpointed Model
        self.model.load_weights(weights_path)
        
        # Remove Masking Layer
        self.model = self.model.layers[-1]
        self.model.trainable = False
        

        
##########################################################################

class ImageSurrogate:
    
    '''
    Wrapper around surrogate model.
    Args:
      surrogate:
    '''
    
    def __init__(self, 
                 original_model, 
                 model_dir = None):
        
        # Models
        self.original_model = original_model
        
        # Parameters Fixed For Images 
        self.input_shape = (224,224,3)
        self.P = 14*14
        self.value_model = ResNet50(
            include_top=False, weights='imagenet', 
            input_shape=self.input_shape, pooling='avg'
        ) 
        self.D = original_model.output.shape[-1]
        
        # model save dir
        if model_dir is None:
            self.save = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.model_dir = os.path.join(os.getcwd(), self.save)
        else:
            self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
              
        
    def train(self, 
              train_data, 
              val_data, 
              max_epochs, 
              batch_size, 
              lookback,
              lr = 1e-3):
        
        # Training Parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lookback = lookback
        
        # Data
        #Check if Provided TF Dataset, if So X should be paired with model predictions
        if (isinstance(train_data, tf.python.data.ops.dataset_ops.PrefetchDataset)              
            or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            @tf.function
            def make_prediction_data(x, y):
                with tf.device("gpu:"+os.environ['CUDA_VISIBLE_DEVICES']):
                    y_model = self.original_model(x)
                return (x, y_model)

            with tf.device("gpu:"+os.environ['CUDA_VISIBLE_DEVICES']):
                train_data = train_data.map(make_prediction_data)
                val_data = val_data.map(make_prediction_data)
            
        else:
            fx_train = self.original_model.predict(train_data)
            fx_val = self.original_model.predict(val_data)
        
        #################################################################
        
        #Make Model w/ Masking
        self.value_model.trainable = True
        
        model_input = Input(shape=self.input_shape, dtype='float64', name='input')
        S = ShapleySampler(self.P, paired_sampling=False, num_samples=1)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32))(S)
        S = Reshape((self.P,))(S)
        S = ResizeMask(in_shape=self.input_shape, mask_size=self.P)(S)
        xs = Multiply()([model_input, S])
        
        net = self.value_model(xs)
        out = Dense(self.D, activation='softmax')(net)
        
        self.model = Model(model_input, out)
        
        # Metrics
        METRICS = [ 
          tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
        ]
        
        # Model Checkpointing
        weights_path = os.path.join(self.model_dir, 'value_weights.h5')
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, 
                                     verbose=1, mode='min', cooldown=1, min_lr=self.lr/100)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=self.lookback) 
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(self.lr)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=OPTIMIZER,
            metrics=METRICS,
        )
        
        # Train Model
        if (isinstance(train_data, tf.python.data.ops.dataset_ops.PrefetchDataset)              
            or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            self.model.fit(x = train_data,
                                 epochs = self.max_epochs,
                                 validation_data = val_data,
                                 callbacks = CALLBACKS)
        else:
            self.model.fit(x = train_data,
                                 y = fx_train,
                                 epochs = self.max_epochs,
                                 batch_size = self.batch_size,
                                 validation_data = (val_data, fx_val),
                                 callbacks = CALLBACKS)
        
        
        # Get Checkpointed Model
        self.model.load_weights(weights_path)
        
        # Remove Masking Layer
        # Remove Masking Layer
        self.model = Sequential(   
            [l for l in self.model.layers[-2:]]
        )
        self.model.trainable = False

##########################################################################
        
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