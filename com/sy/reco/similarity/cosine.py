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
        for a,b in zip(vec1,vec2):
            product+=a*b
            vec1Len+=a**2
            vec2Len+=b**2
        distance=product/np.sqrt(vec1Len*vec2Len)
        return distance
