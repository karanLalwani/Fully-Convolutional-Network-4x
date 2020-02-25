# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:02:18 2020

@author: klal1
"""

import matplotlib.pyplot as plt
import numpy as np

class Metrics:
    def __init__(self, imagePath, annotationPath, model, preprocessingObj):
        self.imagePath = imagePath
        self.annotationPath = annotationPath
        self.model = model
        self.prePro = preprocessingObj

    def getImageFromPrediction(self, prediction):
        return np.argmax(prediction, axis=3).reshape(224, 224)
        
    def plot_predictions(self, images, ):
        _, axs = plt.subplots(3, 3, figsize=(15, 15))
        for n, d in enumerate(self.prePro.data_gen(images, self.imagePath, self.annotationPath, 1)):
            _, h, w, c = d[0].shape
            axs[n][0].imshow(d[0].reshape(h, w, c))
            axs[n][1].imshow(self.getImageFromPrediction(self.model.predict(d[0])))
            axs[n][2].imshow(np.argmax(d[1], axis=3).reshape(h, w))
            if(n == 2):
                break
    
    def plot_graphs(self, typeof, x_label, y_label, title):
        plt.plot(self.model.history.history[typeof])
        plt.plot(self.model.history.history['val_'+typeof])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
    