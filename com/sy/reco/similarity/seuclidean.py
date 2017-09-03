#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import seuclidean

#方差加权距离,标准化欧氏距离,标准化后的值 =  ( 标准化前的值  － 分量的均值 ) /分量的标准差
class Seuclidean():
    def similarity(self,vec1,vec2,variance):
        return seuclidean(vec1,vec2,variance) #后面是一维向量，属性的方差