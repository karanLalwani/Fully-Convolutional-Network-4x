# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:46:15 2020

@author: klal1
"""

from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, add, Conv2DTranspose
from keras.applications.vgg16 import VGG16

class FullyConvolutionalNetwork:
    def __init__(self, image_shape, n_classes):
        self.n_classes = n_classes
        self.image_shape = image_shape

    def get_model(self):
        inp_img = Input(self.image_shape)
        Vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inp_img)
        Vgg16.trainable = False
        x = Vgg16.output
        
        x = Conv2D(4096, kernel_size=7, padding='same', activation='relu', name='d_block6_conv1')(x)
        x = Conv2D(4096, kernel_size=1, padding='same', activation='relu', name='d_block7_conv1')(x)
        x = Conv2D(self.n_classes, kernel_size=1, padding='same', activation='relu', name='d_block8_conv1')(x)
        
        x = UpSampling2D(name='d_block9_pool', interpolation='bilinear')(x)
        skip = Conv2D(self.n_classes, kernel_size=3, padding='same', activation='relu', name='d_block9_conv1')(Vgg16.get_layer('block4_pool').output)
        x = add([x, skip])
    
        x = UpSampling2D(name='d_block10_pool', interpolation='bilinear')(x)
        skip = Conv2D(self.n_classes, kernel_size=3, padding='same', activation='relu', name='d_block10_conv1')(Vgg16.get_layer('block3_pool').output)
        x = add([x, skip])
        
        x = UpSampling2D(size=8, name='d_block11_pool', interpolation='bilinear')(x)
        logits = Conv2D(self.n_classes, kernel_size=3, padding='same', activation='softmax', name='d_block11_conv1')(x)
        
        return Model(inputs=inp_img, outputs=logits)


    def get_modified_model(self):
        inp_img = Input(self.image_shape)
        
        Vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inp_img)
        Vgg16.trainable = False
        x = Vgg16.output
        
        x = Conv2D(4096, kernel_size=7, padding='same', activation='relu', name='d_block6_conv1')(x)
        x = Conv2D(4096, kernel_size=1, padding='same', activation='relu', name='d_block7_conv1')(x)
        x = Conv2D(self.n_classes, kernel_size=1, padding='same', activation='relu', name='d_block8_conv1')(x)
        
        x = Conv2DTranspose(self.n_classes, strides=2, kernel_size=3, padding='same', activation='relu', name='d_block9_conv1')(x)
        skip = Conv2D(self.n_classes, kernel_size=3, padding='same', activation='relu', name='d_block9_conv2')(Vgg16.get_layer('block4_pool').output)
        x = add([x, skip])
    
        x = Conv2DTranspose(self.n_classes, strides=2, kernel_size=3, padding='same', activation='relu', name='d_block10_conv1')(x)
        skip = Conv2DTranspose(self.n_classes, kernel_size=3, padding='same', activation='relu', name='d_block10_conv2')(Vgg16.get_layer('block3_pool').output)
        x = add([x, skip])
        
        x = Conv2DTranspose(self.n_classes, strides=2, kernel_size=3, padding='same', activation='relu', name='d_block11_conv1')(x)
        skip = Conv2DTranspose(self.n_classes, kernel_size=3, padding='same', activation='relu', name='d_block11_conv2')(Vgg16.get_layer('block2_pool').output)
        x = add([x, skip])
        
        logits = Conv2DTranspose(self.n_classes, strides=4, kernel_size=3, padding='same', activation='softmax', name='d_block12_conv1')(x)
      
        return Model(inputs=inp_img, outputs=logits)