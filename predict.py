#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:45:05 2020

@author: Vishal
"""

import numpy as np
from flask import jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('classifier.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        # confidence = round(model.predict_proba(test_image)[0][0],5)
        

        if result[0][0] == 1:
            prediction = 'dog'
            
        else:
            prediction = 'cat'
            

        # output = [{ "image" : prediction, "confidence": str(confidence)}]
        output = [{ "image" : prediction}]
        return output


