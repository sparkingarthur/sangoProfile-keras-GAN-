#coding = utf-8
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import time
import tqdm
import random
import cv2
import PIL.Image as Image

from keras.layers import Input,Dense,Dropout,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Flatten,Activation,UpSampling2D
from keras.layers.merge import concatenate,add,subtract,multiply
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.optimizers import Adam,RMSprop
from keras.models import Model
from GAN_model import get_discriminator_netmowrk,get_generator_network
from sklearn.utils import shuffle as sklearn_shuffle


import matplotlib.gridspec as gridspec

# np.random.seed(2018)


samplePath = '../resizedpic'  #named from 1.jpg to N.jpg
image_save_dir='../image_save_dir/'

logPath = '../log'
modelsavePath = '../savedmodel/'

imageShape = (64,64,3)
fakeShpe = (1,1,2048)
TOTAL_SAMPLES = 1198
NUM_STEPS=20000
BATCH_SIZE = 64

WRITE_LOG = True

'''
 :return a number between [0,1]
'''
def norm_pic(pic):
    img = pic /255.
    return img

'''
  :return a number between [0,255]
'''
def denorm_pic(pic):
    img = pic*255
    return img.astype(np.uint8)

def sample_real_X(fileLists,image_shape=(64,64,3),batch_size=32):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    selected = random.sample(fileLists,batch_size)
    #print(selected)
    # sample_names = [os.path.join(samplePath,x) for x in selected]
    for i,file in enumerate(selected):
        file_path = os.path.join(samplePath,file)
        # print(file_name)
        img = cv2.imread(file_path)
        img = norm_pic(img)
        sample[i]=img
    #print(sample.shape)
    return sample

def gen_fake_seed(batch_size=32,noise_shape=(1,1,2048)):
    seeds = np.random.random(size=(batch_size,) + noise_shape)
    #print(seeds)
    return seeds

def generate_fake_imgs(generator, save_dir,save_file_name):
    random_seed = gen_fake_seed(batch_size=32)
    fake_data_X = generator.predict(random_seed)
    print("Displaying generated images")
    plt.figure(figsize=(4, 4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0], 16, replace=False)
    for i in range(16):
        # plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :, :, :]
        fig = plt.imshow(denorm_pic(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir + save_file_name+".png", bbox_inches='tight', pad_inches=0)
    #plt.show()

def save_imgs(imgs,save_dir):
    plt.figure(figsize=(4, 4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(imgs.shape[0], 16, replace=False)
    # print(rand_indices)
    for i in range(16):
        # plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = imgs[rand_index, :, :, :]
        fig = plt.imshow(denorm_pic(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)
    #plt.show()


if __name__ == '__main__':
    files = os.listdir(samplePath)

    generator = get_generator_network(fakeinputs_shape=fakeShpe)
    discriminator = get_discriminator_netmowrk(inputs_shape=imageShape)
    discriminator.trainable = False
    opt = RMSprop(lr=0.0001)  # same as gen
    gen_inp = Input(shape=fakeShpe)
    GAN_inp = generator(gen_inp)
    GAN_opt = discriminator(GAN_inp)
    gan_model = Model(inputs=gen_inp, outputs=GAN_opt)
    gan_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    gan_model.summary()
    text_file = None
    if WRITE_LOG:
        text_file = open(logPath + "/training_log.txt", "w+")
    for step in range(NUM_STEPS):
        print("Begin step: ", step)

        avg_disc_fake_loss=[]
        avg_GAN_loss = []

        step_begin_time = time.time()
        ### ==============prepare for the data=======================
        noise_seeds = gen_fake_seed(batch_size=BATCH_SIZE,noise_shape=fakeShpe)
        fake_data_X = generator.predict(noise_seeds)
        real_data_X = sample_real_X(fileLists=files, image_shape=imageShape, batch_size=BATCH_SIZE)
        if((step+1) % 100 == 0):
            save_imgs(fake_data_X,image_save_dir+'step'+str(step)+'.png') ## save the results of generator every 20 steps

        # real_data_Y = np.ones(BATCH_SIZE)
        real_data_Y = np.ones(BATCH_SIZE) - np.random.random_sample(BATCH_SIZE)*0.2
        # fake_data_Y = np.zeros(BATCH_SIZE)
        fake_data_Y = np.random.random_sample(BATCH_SIZE)*0.2

        dis_data_X = np.concatenate([real_data_X, fake_data_X])
        dis_data_Y = np.concatenate([real_data_Y,fake_data_Y])

        ### shuffle the training data for training
        dis_data_X,dis_data_Y = sklearn_shuffle(dis_data_X,dis_data_Y)
        ### ==============training begins==================
        ### step1:fix the generator, train the discriminator
        discriminator.trainable = True
        generator.trainable = False
        dis_metrics = discriminator.train_on_batch(dis_data_X,dis_data_Y)
        print("step %d: discriminator loss %.6f, acc %.6f"%(step,dis_metrics[0],dis_metrics[1]))
        avg_disc_fake_loss.append(dis_metrics[0])
        ### step2: fix the discriminator ,train the generator
        gan_data_X = gen_fake_seed(batch_size=BATCH_SIZE,noise_shape=fakeShpe)
        gan_data_Y = real_data_Y
        generator.trainable = True
        discriminator.trainable = False
        gan_metrics = gan_model.train_on_batch(gan_data_X, gan_data_Y)
        print("step %d: gan loss %.6f, acc %.6f" % (step, gan_metrics[0], gan_metrics[1]))
        avg_GAN_loss.append(gan_metrics[0])
        if WRITE_LOG:
            text_file.write("step %d: dis loss %.6f, dis acc %.6f, gan loss %.6f, gan acc %.6f\r\n"
                            % (step, dis_metrics[0], dis_metrics[1],gan_metrics[0], gan_metrics[1]))

        if ((step + 1) % 500) == 0:
            print("-----------------------------------------------------------------")
            print("Average Discriminator loss: %f" % (np.mean(avg_disc_fake_loss)))
            print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
            print("-----------------------------------------------------------------")
            discriminator.trainable = True
            generator.trainable = True
            generator.save(modelsavePath + str(step) + "_GEN_weights_and_arch.hdf5")
            discriminator.save(modelsavePath + str(step) + "_DISC_weights_and_arch.hdf5")
        end_time = time.time()
        diff_time = int(end_time - step_begin_time)
        print("Step %d completed. Time took: %s secs." % (step, diff_time))
    if WRITE_LOG:
        text_file.close()
    for i in range(10):
        generate_fake_imgs(generator, image_save_dir,'fakefinal'+str(i))

    #generate_fake_imgs(generator,'./','testgen')
    #sample_real_X(files)
    #gen_fake_seed()
    # img = cv2.imread('../resizedpic/1.jpg')
    #print(type(img))
