#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

from com.sy.reco.util.dataprocess import DataProcess
from com.sy.reco.util.readrating import ReadRating
from com.sy.reco.recommendation.userbased import UserBased
from com.sy.reco.recommendation.itembased import ItemBased
from com.sy.reco.recommendation.slopeone import SlopeOne
from com.sy.reco.similarity.cosine import Cosine

class Test():
    #数据预处理
    def dataProcess(self,ratingPath,itemPath,savetoPath):
        DataProcess(ratingPath,itemPath,savetoPath)
    #读取评分数据
    def readRatingMatrix(self,ratingMatrixPath):
        read=ReadRating()
        ratingMatrix=read.readRatingData(ratingMatrixPath)
        return ratingMatrix.values[:,1:]#去除第一列是用户的标号，返回的是numpy中的ndarray类型
    #基于user的预测
    def userBased(self,ratingMatrix,user_vec):
        userBased=UserBased(ratingMatrix,user_vec)
        cosine=Cosine()
        k_simNeighbours=userBased.kNeighbours(10,cosine)
        predictRating=userBased.getRatingPredict(k_simNeighbours)
        print(predictRating)
    #基于item预测
    def itemBased(self,ratingMatrix,user_vec):
        itemBased=ItemBased(ratingMatrix,user_vec)
        cosine=Cosine()
        itemBased.itemSimilarityMatrix(cosine)
        predictRating=itemBased.getRatingPredict(10)
        print(predictRating)
    #slopeone预测
    def slopeOne(self,ratingMatrix,user_vec):
        slopeone=SlopeOne(ratingMatrix,user_vec)
        slopeone.itemDeviationMatrix()
        predictRating=slopeone.getRatingPredict(10)
        print(predictRating)

if __name__=='__main__':
    test=Test()
    #test.dataProcess('G:\\python workspace\\recommendation_methods\\data\\ratings.csv','G:\\python workspace\\recommendation_methods\\data\\movies.csv','G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv')
    ratingMatrix=test.readRatingMatrix('G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv')
    #test.userBased(ratingMatrix,ratingMatrix[0])#第一个用户测试
    #test.itemBased(ratingMatrix,ratingMatrix[0])
    test.slopeOne(ratingMatrix,ratingMatrix[0])



