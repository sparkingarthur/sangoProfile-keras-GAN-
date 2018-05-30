#coding = utf-8
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import time
import tqdm
import random

from keras.layers import Input,Dense,Dropout,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Flatten,Activation,UpSampling2D
from keras.layers.merge import concatenate,add,subtract,multiply
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.optimizers import Adam,RMSprop
from keras.models import Model



'''
  get_discriminator_netmowrk:used to be a discriminator
  @fakeinputs_shape:input shape of one picture 
  @kernel_init:init_method of the layers
  @return: a discriminator model
'''
def get_discriminator_netmowrk(inputs_shape=(64,64,3),kernel_init = 'glorot_uniform'):
    dropout_prob = 0.4

    # kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    dis_input = Input(shape=inputs_shape)

    discriminator = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)
    # discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)

    # discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    # discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)

    # discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    # discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)

    # discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    # discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)

    discriminator = Flatten()(discriminator)
    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)

    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input=dis_input, output=discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    discriminator_model.summary()
    return discriminator_model

'''
  get_generator_network:used to be a generator
  @fakeinputs_shape:generator input shape
  @kernel_init:init_method of the layers
  @return: a generator model
'''
def get_generator_network(fakeinputs_shape=(1,1,2048),kernel_init = 'glorot_uniform'):
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    # kernel_init = RandomNormal(mean=0.0, stddev=0.01)

    gen_input = Input(shape=fakeinputs_shape)  # if want to directly use with conv layer next
    # gen_input = Input(shape = [noise_shape]) #if want to use with dense layer next

    generator = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding="valid",
                                data_format="channels_last", kernel_initializer=kernel_init)(gen_input)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # generator = bilinear2x(generator,256,kernel_size=(4,4))
    # generator = UpSampling2D(size=(2, 2))(generator)
    # generator = SubPixelUpscaling(scale_factor=2)(generator)
    # generator = Conv2D(filters = 256, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last", kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # generator = bilinear2x(generator,128,kernel_size=(4,4))
    # generator = UpSampling2D(size=(2, 2))(generator)
    # generator = SubPixelUpscaling(scale_factor=2)(generator)
    # generator = Conv2D(filters = 128, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last", kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # generator = bilinear2x(generator,64,kernel_size=(4,4))
    # generator = UpSampling2D(size=(2, 2))(generator)
    # generator = SubPixelUpscaling(scale_factor=2)(generator)
    # generator = Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last", kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last",
                       kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # generator = bilinear2x(generator,3,kernel_size=(3,3))
    # generator = UpSampling2D(size=(2, 2))(generator)
    # generator = SubPixelUpscaling(scale_factor=2)(generator)
    # generator = Conv2D(filters = 3, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last", kernel_initializer=kernel_init)(generator)
    generator = Activation('tanh')(generator)
    # generator = Activation('sigmoid')(generator)

    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(inputs=gen_input, outputs=generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model


#get_discriminator_netmowrk()
# get_generator_network()