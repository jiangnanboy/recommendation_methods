#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.spatial.distance import cityblock

#曼哈顿
class Manhattan():
    def similarity(self,vec1,vec2):
        return cityblock(vec1,vec2)