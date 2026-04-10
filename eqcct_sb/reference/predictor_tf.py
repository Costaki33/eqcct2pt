import os
import gc 
import sys

# # Save the original stderr (so we can restore it later if needed)
# original_stderr = sys.stderr

# # Open /dev/null to discard unwanted logs
# devnull = open(os.devnull, 'w')

# # Redirect only TensorFlow-related logs to /dev/null
# os.dup2(devnull.fileno(), sys.stderr.fileno())

import ast
import ray
import csv
import time
import glob
import queue
import shutil
import psutil
import random
import logging
import platform
import numpy as np
import pandas as pd
from os import listdir
from io import StringIO
from contextlib import redirect_stdout
from datetime import datetime, timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import warnings
import multiprocessing
import absl.logging
from silence_tensorflow import silence_tensorflow

# Silence TensorFlow logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
warnings.filterwarnings("ignore")
silence_tensorflow()

def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

 
def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+tf.keras.backend.epsilon()))


def wbceEdit( y_true, y_pred) :
    ms = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)) 
    ssim = 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
    return (ssim + ms)

w1 = 6000
w2 = 3
drop_rate = 0.2
stochastic_depth_rate = 0.1

positional_emb = False
conv_layers = 4
num_classes = 1
input_shape = (w1, w2)
num_classes = 1
input_shape = (6000, 3)
image_size = 6000  # We'll resize input images to this size
patch_size = 40  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 40

num_heads = 4
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = tf.keras.layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    


def convF1(inpt, D1, fil_ord, Dr):

    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    #filters = inpt._keras_shape[channel_axis]
    filters = int(inpt.shape[-1])
    
    #infx = tf.keras.layers.Activation(tf.nn.gelu')(inpt)
    pre = tf.keras.layers.Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inpt)
    pre = tf.keras.layers.BatchNormalization()(pre)    
    pre = tf.keras.layers.Activation(tf.nn.gelu)(pre)
    
    #shared_conv = tf.keras.layers.Conv1D(D1,  fil_ord, strides =(1), padding='same')
    
    inf = tf.keras.layers.Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(pre)
    inf = tf.keras.layers.BatchNormalization()(inf)    
    inf = tf.keras.layers.Activation(tf.nn.gelu)(inf)
    inf = tf.keras.layers.Add()([inf,inpt])
    
    inf1 = tf.keras.layers.Conv1D(D1,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inf)
    inf1 = tf.keras.layers.BatchNormalization()(inf1)  
    inf1 = tf.keras.layers.Activation(tf.nn.gelu)(inf1)    
    encode = tf.keras.layers.Dropout(Dr)(inf1)

    return encode


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        #x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = tf.keras.layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded


def create_cct_modelP(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        #encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        #attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def create_cct_modelS(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def load_eqcct_model(input_modelP, input_modelS, log_file="results/logs/model.log"):
    # print(f"[{datetime.now()}] Loading EQCCT model.")
    
    # with open(log_file, mode="w", buffering=1) as log:
    #     log.write(f"*** Loading the model ...\n")

    # Model CCT
    inputs = tf.keras.layers.Input(shape=input_shape,name='input')

    featuresP = create_cct_modelP(inputs)
    featuresP = tf.keras.layers.Reshape((6000,1))(featuresP)

    featuresS = create_cct_modelS(inputs)
    featuresS = tf.keras.layers.Reshape((6000,1))(featuresS)

    logitp  = tf.keras.layers.Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_P')(featuresP)
    logits  = tf.keras.layers.Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresS)

    modelP = tf.keras.models.Model(inputs=[inputs], outputs=[logitp])
    modelS = tf.keras.models.Model(inputs=[inputs], outputs=[logits])

    model = tf.keras.models.Model(inputs=[inputs], outputs=[logitp,logits])

    def _load_branch_weights(branch: tf.keras.models.Model, path: str, label: str) -> str:
        """
        Try strict name-based load first (avoids permuting layers vs HDF5 order).
        Keras 3 can reject older MHA checkpoints with reshape/axis errors; then try
        skip_mismatch, then fall back to positional load (least reliable).
        """
        attempts = [
            ("by_name strict", {"by_name": True, "skip_mismatch": False}),
            ("by_name skip_mismatch", {"by_name": True, "skip_mismatch": True}),
            ("positional", {}),
        ]
        last_err = None
        for desc, kw in attempts:
            if not kw:
                try:
                    branch.load_weights(path)
                    return desc
                except Exception as e:
                    last_err = e
                    continue
            try:
                branch.load_weights(path, **kw)
                return desc
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Could not load weights for {label} from {path!r}: {last_err!r}"
        ) from last_err

    p_how = _load_branch_weights(modelP, input_modelP, "modelP")
    s_how = _load_branch_weights(modelS, input_modelS, "modelS")

    sgd = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=sgd,
        loss=["binary_crossentropy", "binary_crossentropy"],
        metrics=["acc", f1, precision, recall],
    )

    if p_how != "by_name strict" or s_how != "by_name strict":
        # warnings.filterwarnings("ignore") is set globally in this module; use stderr.
        print(
            f"[eqcct predictor_tf] weight load strategy: P={p_how!r}, S={s_how!r} "
            f"(if not 'by_name strict', verify TF outputs against a known-good Keras version).",
            file=sys.stderr,
        )

    # log.write(f"*** Loading is complete!")

    return modelP, modelS