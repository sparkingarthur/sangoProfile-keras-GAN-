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
    dis_input = Input(shape=inputs_shape)
    discriminator = Conv2D(filters=32,kernel_size=(5,5),padding='same',kernel_initializer=kernel_init)(dis_input)
    discriminator = BatchNormalization()(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = MaxPooling2D()(discriminator)
    discriminator = Conv2D(filters=64,kernel_size=(4,4),padding='valid')(discriminator)
    discriminator = BatchNormalization()(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = MaxPooling2D()(discriminator)
    discriminator = Conv2D(filters=128,kernel_size=(3,3),padding='valid')(discriminator)
    discriminator = BatchNormalization()(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = MaxPooling2D()(discriminator)
    discriminator = Conv2D(filters=256, kernel_size=(2,2), padding='valid')(discriminator)
    discriminator = BatchNormalization()(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = MaxPooling2D()(discriminator)
    discriminator = Conv2D(filters=512, kernel_size=(1,1), padding='valid')(discriminator)
    discriminator = BatchNormalization()(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = Flatten()(discriminator)

    # discriminator = MinibatchDiscrimination(100,5)(discriminator)
    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)

    dis_opt = RMSprop(lr=0.0001)
    discriminator_model = Model(inputs=dis_input, outputs=discriminator)
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
    gen_input = Input(shape=fakeinputs_shape)
    generator = Conv2DTranspose(filters=512, kernel_size=(1, 1), #deconv
                                padding="valid", kernel_initializer=kernel_init)(gen_input)
    generator = BatchNormalization()(generator)
    generator = PReLU()(generator)
    generator = UpSampling2D()(generator)

    generator = Conv2DTranspose(filters=256, kernel_size=(2, 2), padding="valid",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization()(generator)
    generator = PReLU()(generator)
    generator = UpSampling2D()(generator)

    generator = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="valid",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization()(generator)
    generator = PReLU()(generator)
    generator = UpSampling2D()(generator)

    generator = Conv2DTranspose(filters=64, kernel_size=(4, 4), padding="same",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization()(generator)
    generator = PReLU()(generator)
    generator = UpSampling2D()(generator)

    generator = Conv2DTranspose(filters=32, kernel_size=(5, 5), padding="same",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization()(generator)
    generator = PReLU()(generator)
    generator = UpSampling2D()(generator)

    generator = Conv2DTranspose(filters=3, kernel_size=(1, 1), padding="same",
                                kernel_initializer=kernel_init)(generator)

    gen_opt = RMSprop(lr=0.0001)
    generator_model = Model(inputs=gen_input, outputs=generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model


#get_discriminator_netmowrk()
# get_generator_network()