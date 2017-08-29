#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import cosine

#余弦
class Cosine():
    def similarity(self,vec1,vec2):
        return 1-cosine(vec1,vec2)