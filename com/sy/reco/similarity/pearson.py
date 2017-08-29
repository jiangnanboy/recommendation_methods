#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from scipy.stats  import pearsonr

#皮尔逊
class Pearson():
     def similarity(self,vec1,vec2):
         return pearsonr(vec1,vec2)[0]