#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import sqeuclidean

#平方欧氏
class Sqeuclidean():
    def similarity(self,vec1,vec2):
        return sqeuclidean(vec1,vec2)