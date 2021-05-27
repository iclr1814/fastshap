from shap.utils._legacy import convert_to_link
from shap.utils._legacy import Link

import numpy as np
import pandas as pd
import scipy as sp
import math

from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Lambda, Dropout, Multiply, BatchNormalization, Reshape, Concatenate, TimeDistributed, Conv2D, Permute, UpSampling3D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from datetime import datetime
import os

from utils import convert_to_linkTF, ShapleySampler, ResizeMask

from tensorflow.keras.applications.resnet50 import ResNet50


##########################################################################

class FastSHAP:

    def __init__(self, 
                 imputer,
                 explainer_model,
                 normalization,
                 model_dir = None, 
                 link='logit'):

        # Link
        self.link = convert_to_link(link)
        self.linkTF = convert_to_linkTF(link)
        self.linkfv = np.vectorize(self.link.f)
        
        # Models
        self.imputer = imputer
        self.explainer_model = explainer_model
        
        # Game Parameters
        self.P = imputer.input.shape[-1]
        self.D = imputer.output.shape[-1]
        
        # Null
        self.null = np.squeeze(imputer.predict(np.zeros((1, self.P))))
        
        # Set up normalization.
        if normalization is None or normalization=='additive':
            self.normalization = normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))
        
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
              num_samples,
              lr = 1e-3,
              paired_sampling = True, 
              eff_lambda = 0,
              verbose = 0,
              lookback=20):
        
        #Training Parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lookback = lookback
        
        # Dummy Labels
        y_train_dummy = np.zeros((train_data.shape[0],1))
        y_val_dummy = np.zeros((val_data.shape[0],1))
        
        # Subset Sampling Hyperparameters
        self.paired_sampling = paired_sampling
        self.num_samples = num_samples
        
        # Set up normalization.
        self.eff_lambda = eff_lambda
        
        ###### Create Model ######
        model_input = Input(shape=self.P, dtype='float32', name='input')
        S = ShapleySampler(self.P, paired_sampling=self.paired_sampling, num_samples = self.num_samples)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32), name='S')(S)
        
        #If Paired Double num_samples:
        if self.paired_sampling:
            num_samples = 2 * self.num_samples
        else:
            num_samples = self.num_samples
        
        # Learn Phi 
        phi = self.explainer_model(model_input) 
        
        #Efficency Normalization
        gap = Lambda(lambda x: 
                     (self.linkTF.f(K.stop_gradient(K.clip(self.imputer(x[0]), 1e-7, 1-1e-7))) -  
                      self.linkTF.f(tf.constant(self.null, dtype=tf.float32))) -  
                     K.sum(x[1], -1)
                    )([model_input, phi])
        if self.normalization == 'additive':
            phi = Lambda(lambda x: 
                         x[1] + tf.expand_dims(x[0]/self.P, -1)
                        )([gap, phi])
        
        # Name Output Layer and Reshape
        phi = Layer(name='phi')(phi)
        phi = Reshape((self.P*self.D,))(phi)
        
        # Repeat Phi for Multiple Subset Sampling
        phi_repeat = tf.keras.layers.RepeatVector(num_samples)(phi)
        phi_repeat = Reshape((num_samples, self.D, self.P),  name='phi_repeat')(phi_repeat)
        
        # Calculate output 
        phi_S = Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], 2)], 2))([phi_repeat, S])
        out = TimeDistributed(
            Lambda(lambda x: 
                   tf.squeeze(tf.matmul(x[:,:self.D,:], tf.expand_dims(x[:,-1,:], -1)), -1)),
            name = 'linear_model'
        )(phi_S)
        
        # Repeat Input for Multiple Subset Sampling
        model_input_repeat = Reshape((1, self.P))(model_input)
        model_input_repeat = tf.keras.layers.UpSampling1D(size=num_samples, name='model_input_repeat')(model_input_repeat)

        # yAdj = link(f(x_s))- link(E[f(x)])
        xs = Multiply()([model_input_repeat, S])
        f_xs = TimeDistributed(self.imputer, name='f_xs')(xs)
        yAdj = TimeDistributed(
            Lambda(lambda x: K.stop_gradient(
                self.linkTF.f(K.clip(x, 1e-7, 1-1e-7)) - self.linkTF.f(tf.constant(self.null, dtype=tf.float32))
            )), name = 'yAdj'
        )(f_xs)
        
        ## MSE Loss w/ L1 Regularization
        SHAPloss = tf.reduce_mean(tf.keras.losses.MSE(yAdj, out))
        EFFloss = self.eff_lambda*tf.reduce_mean(gap**2) 
 
        self.explainer = Model(model_input, out)
        
        self.explainer.add_loss(SHAPloss)
        self.explainer.add_loss(EFFloss)
        
        self.explainer.add_metric(SHAPloss, name='shap_loss', aggregation='mean')
        self.explainer.add_metric(EFFloss, name='eff_loss', aggregation='mean')
        
        
        # Model Checkpointing
        explainer_weights_path = os.path.join(self.model_dir, 'explainer_weights.h5')
        checkpoint = ModelCheckpoint(explainer_weights_path, monitor='val_shap_loss', verbose=verbose, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_shap_loss", mode="min", patience=self.lookback) 
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop]
        OPTIMIZER = tf.keras.optimizers.Adam(self.lr)
        
        self.explainer.compile(
            optimizer=OPTIMIZER
        )
        
        # Train Model
        history = self.explainer.fit(x = train_data, 
                                     y = y_train_dummy, 
                                     epochs = self.max_epochs,
                                     batch_size = self.batch_size,
                                     validation_data = (val_data, 
                                                        y_val_dummy),
                                     callbacks = CALLBACKS,
                                     verbose=verbose)
        
        self.val_losses = history.history['val_shap_loss']
        
        # Get Checkpointed Model
        self.explainer.load_weights(explainer_weights_path)
        
        # Extract Explainer
        self.explainer = Model(self.explainer.get_layer('input').input, 
                                     self.explainer.get_layer('phi').output)
        self.explainer.trainable = False
        
        
    def shap_values(self, X):
        """ cite: https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
        """

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            X = X.values

        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tsolil()
        assert x_type.endswith(arr_type) or sp.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        # single instance
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
            explanation = self.explainer.predict(X)[0]
            
            # efficeny normalization
            if self.normalization == 'additive':
                prediction = self.linkfv(self.imputer.model.predict(X)[0]) - self.linkfv(self.null)
                diff = (prediction - explanation.sum(-1))
                explanation += np.expand_dims(diff/explanation.shape[-1], -1)

            # vector-output
            s = explanation.shape
            if len(s) == 2:
                outs = [np.zeros(s[1]) for j in range(s[0])]
                for j in range(s[0]):
                    outs[j] = explanation[j, :]

            # single-output
            else:
                out = np.zeros(s[1])
                out[:] = explanation[0]
                
            
        # explain the whole dataset
        elif len(X.shape) == 2:
            explanations = self.explainer.predict(X)
            
            # efficeny normalization
            if self.normalization == 'additive':
                prediction = self.linkfv(self.imputer.model.predict(X)) - self.linkfv(self.null)
                diff = (prediction - explanations.sum(-1))
                explanations += np.expand_dims(diff/explanations.shape[-1], -1)

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[1])) for j in range(s[0])]
                for i in range(X.shape[0]):
                    for j in range(s[0]):
                        outs[j][i] = explanations[i][j, :]
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[1]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i][0]
                return out
        

##########################################################################        
        
class ImageFastSHAP:

    def __init__(self, 
                 imputer,
                 normalization,
                 model_dir = None, 
                 link='logit'):

        # Link
        self.link = convert_to_link(link)
        self.linkTF = convert_to_linkTF(link)
        self.linkfv = np.vectorize(self.link.f)
        
        # Models
        self.imputer = imputer
        
        # Parameters Fixed for Images
        self.input_shape = (224,224,3)
        self.P = 14*14
        self.D = imputer.output.shape[-1]
        
        # Null
        self.null = np.squeeze(imputer.predict(np.zeros(tuple([1]+list(self.input_shape)))))
        
        # Set up normalization.
        if normalization is None or normalization=='additive':
            self.normalization = normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))
        
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
              num_samples,
              lr = 1e-3,
              paired_sampling = True, 
              eff_lambda = 0,
              verbose = 0,
              lookback=20):
        
        #Training Parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lookback = lookback
        
        # Data (get dummy labels)
        if not (isinstance(train_data, tf.python.data.ops.dataset_ops.PrefetchDataset)              
            or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            y_train_dummy = np.zeros((train_data.shape[0],1))
            y_val_dummy = np.zeros((val_data.shape[0],1))
        
        # Subset Sampling Hyperparameters
        self.paired_sampling = paired_sampling
        self.num_samples = num_samples
        
        # Set up normalization.
        self.eff_lambda = eff_lambda
        
        ##########################################################################
        
        ###### Create Model ######
        model_input = Input(shape=self.input_shape, dtype='float32', name='input')
        S = ShapleySampler(self.P, paired_sampling=self.paired_sampling, num_samples = self.num_samples)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32), name='S')(S)
        
        #If Paired Double num_samples:
        if self.paired_sampling:
            num_samples = 2 * self.num_samples
        else:
            num_samples = self.num_samples
        
        #Phi Model
        base_model = ResNet50(
            include_top=False, weights='imagenet', 
            input_shape=self.input_shape
        )
        base_model = Model(base_model.input, base_model.get_layer('conv4_block3_2_conv').output)
        base_model.trainable = True

        net = base_model(model_input)
        
        # Learn Phi 
        phi = Conv2D(self.D, 1)(net)
        phi = Reshape((self.P, self.D))(phi)
        phi = Permute((2,1))(phi)
        
        #Efficency Normalization
        gap = Lambda(lambda x: 
                     (self.linkTF.f(K.stop_gradient(K.clip(self.imputer(x[0]), 1e-7, 1-1e-7))) -  
                      self.linkTF.f(tf.constant(self.null, dtype=tf.float32))) -  
                     K.sum(x[1], -1)
                    )([model_input, phi])
        if self.normalization == 'additive':
            phi = Lambda(lambda x: 
                         x[1] + tf.expand_dims(x[0]/self.P, -1)
                        )([gap, phi])
        
        # Name Output Layer and Reshape
        phi = Layer(name='phi')(phi)
        phi = Reshape((self.P*self.D,))(phi)
        
        # Repeat Phi for Multiple Subset Sampling
        phi_repeat = tf.keras.layers.RepeatVector(num_samples)(phi)
        phi_repeat = Reshape((num_samples, self.D, self.P),  name='phi_repeat')(phi_repeat)
        
        # Calculate output 
        phi_S = Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], 2)], 2))([phi_repeat, S])
        out = TimeDistributed(
            Lambda(lambda x: 
                   tf.squeeze(tf.matmul(x[:,:self.D,:], tf.expand_dims(x[:,-1,:], -1)), -1)),
            name = 'linear_model'
        )(phi_S)
        
        # Repeat Input for Multiple Subset Sampling
        model_input_repeat = Reshape((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))(model_input)
        model_input_repeat = UpSampling3D(size=(num_samples, 1, 1), name='model_input_repeat')(model_input_repeat)
        
        # Resize Masks
        S_RM = TimeDistributed(ResizeMask(in_shape=self.input_shape, mask_size=self.P), name='S_RM')(S)

        # yAdj = link(f(x_s))- link(E[f(x)])
        xs = Multiply()([model_input_repeat, S_RM])
        f_xs = TimeDistributed(self.imputer, name='f_xs')(xs)
        yAdj = TimeDistributed(
            Lambda(lambda x: K.stop_gradient(
                self.linkTF.f(K.clip(x, 1e-7, 1-1e-7)) - self.linkTF.f(tf.constant(self.null, dtype=tf.float32))
            )), name = 'yAdj'
        )(f_xs)
        
        ## MSE Loss w/ Efficiency Regularization         
        SHAPloss = tf.reduce_mean(tf.keras.losses.MSE(yAdj, out))
        EFFloss = self.eff_lambda*tf.reduce_mean(gap**2) 
 
        self.explainer = Model(model_input, out)
        
        self.explainer.add_loss(SHAPloss)
        self.explainer.add_loss(EFFloss)
        
        self.explainer.add_metric(SHAPloss, name='shap_loss', aggregation='mean')
        self.explainer.add_metric(EFFloss, name='eff_loss', aggregation='mean')
        
        # Model Checkpointing
        explainer_weights_path = os.path.join(self.model_dir, 'explainer_weights.h5')
        checkpoint = ModelCheckpoint(explainer_weights_path, monitor='val_shap_loss', verbose=verbose, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_shap_loss", mode="min", patience=self.lookback) 
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_shap_loss', factor=0.8, patience=3, 
                                     verbose=1, mode='min', cooldown=1, min_lr=1e-6)
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(self.lr)
        
        self.explainer.compile(
            optimizer=OPTIMIZER
        )
        
        # Train Model
        if (isinstance(train_data, tf.python.data.ops.dataset_ops.PrefetchDataset)              
            or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            history = self.explainer.fit(x = train_data,
                                               epochs = self.max_epochs,
                                               validation_data = val_data,
                                               callbacks = CALLBACKS, 
                                               verbose=verbose)
        else:
            history = self.explainer.fit(x = train_data,
                                               y = y_train_dummy,
                                               epochs = self.max_epochs,
                                               batch_size = self.batch_size,
                                               validation_data = (val_data, y_val_dummy),
                                               callbacks = CALLBACKS, 
                                               verbose=verbose)

        self.val_losses = history.history['val_shap_loss']
        
        # Get Checkpointed Model
        self.explainer.load_weights(explainer_weights_path)
        
        
        #  Extract Explainer
        # 1) Get Base Model for Phi
        base_model = Model(self.explainer.get_layer('input').input, 
                           self.explainer.get_layer('phi').output)
        base_model.summary()
        # 2) Resize
        model_input = Input(shape=self.input_shape, dtype='float32', name='input')

        phi = base_model(model_input)
        phi = Permute((2,1))(phi)
        phi = ResizeMask(in_shape=self.input_shape, mask_size=self.P, output_channels=self.D)(phi)

        self.explainer = Model(model_input, phi)
        self.explainer.trainable = False
        
        
    def shap_values(self, X):

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            X = X.values

        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tsolil()
        assert x_type.endswith(arr_type) or sp.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type
        assert len(X.shape) == 3 or len(X.shape) == 4, "Instance must have 1 or 2 dimensions!"

        # single instance
        if len(X.shape) == 3:
            X = np.expand_dims(X, 0)
            explanation = self.explainer.predict(X)[0]
            
            # vector-output
            out = [explanations[:,:,i].numpy() for i in range(self.D)]
                 
        # explain the whole dataset
        elif len(X.shape) == 4:
            explanations = self.explainer.predict(X)
            
            # vector-output
            out = [explanations[:,:,:,i].numpy() for i in range(self.D)]
            
        return out
        

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