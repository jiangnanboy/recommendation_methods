#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from com.sy.reco.util.dataprocess import DataProcess

class Test():

    def dataProcess(self,ratingPath,itemPath,savetoPath):
        DataProcess(ratingPath,itemPath,savetoPath)

if __name__=='__main__':
    test=Test()
    #test.dataProcess('G:\\python workspace\\recommendation_methods\\data\\ratings.csv',
                     #'G:\\python workspace\\recommendation_methods\\data\\movies.csv',
                     #'G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv')



