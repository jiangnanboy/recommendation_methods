#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import cityblock

#曼哈顿
class Manhattan():
    #def similarity(self,vec1,vec2):
        #return cityblock(vec1,vec2)

    def similarity(self,vec1,vec2):
        distance=0.0
        for i in range(len(vec1)):
            distance+=abs(vec1[i]-vec2[i])
        return distance
