#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import cosine
import numpy as np

#余弦
class Cosine():
    #def similarity(self,vec1,vec2):
        #return 1-cosine(vec1,vec2)

    def similarity(self,vec1,vec2):
        distance=0.0
        product=0.0
        vec1Len=0.0
        vec2Len=0.0
        for i in range(len(vec1)):
            product+=vec1[i]*vec2[i]
            vec1Len+=vec1[i]**2
            vec2Len+=vec2[i]**2
        distance=product/np.sqrt(vec1Len)*np.sqrt(vec2Len)
        return distance