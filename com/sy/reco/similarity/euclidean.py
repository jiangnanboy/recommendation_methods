#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import euclidean

#欧氏
class Euclidean():
    def similarity(self,vec1,vec2):
        return euclidean(vec1,vec2)