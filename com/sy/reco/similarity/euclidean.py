#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import euclidean
import numpy as np

#欧氏
class Euclidean():
    #def similarity(self,vec1,vec2):
        #return euclidean(vec1,vec2)

    def similarity(self,vec1,vec2):
        distance=0.0
        for i in range(len(vec1)):
            if vec1[i]==0 or vec2[i]==0:
                continue
            distance+=(vec1[i]-vec2[i])**2
        distance=np.sqrt(distance)
        return distance