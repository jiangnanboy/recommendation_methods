#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np

from com.sy.reco.util.dataprocess import DataProcess
from com.sy.reco.util.readrating import ReadRating
from com.sy.reco.recommendation.userbased import UserBased
from com.sy.reco.recommendation.itembased import ItemBased
from com.sy.reco.recommendation.slopeone import SlopeOne
from com.sy.reco.recommendation.matrix_factorization.lfm import LFM
from com.sy.reco.recommendation.matrix_factorization.biaslfm import BiasLFM
from com.sy.reco.recommendation.matrix_factorization.svdpp import SVDPP
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
    #LFM隐语义矩阵分解
    def lfm_rec(self,ratingMatrix,F,α,λ,max_iter):
        lfm=LFM(ratingMatrix,F,α,λ)
        predictRating=lfm.iteration_train(max_iter)
        #predictRating=lfm.convergence_train(0.1)
        print(predictRating)
    #加入偏置项的lfm
    def biaslfm(self,ratingMatrix,F,α,λ,max_iter):
        biasLFm=BiasLFM(ratingMatrix,F,α,λ)
        predictRating=biasLFm.iteration_train(max_iter)
        print(predictRating)
    #SVDPP
    def svdpp(self,ratingMatrix,F,α,λ,max_iter):
        svdPP=SVDPP(ratingMatrix,F,α,λ)
        predictRating=svdPP.iteration_train(max_iter)
        print(predictRating)

if __name__=='__main__':
    test=Test()
    ratingMatrix = test.readRatingMatrix('G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv')
    recType=6
    if recType==1:#userBased
        test.userBased(ratingMatrix,ratingMatrix[0])#第一个用户测试
    elif recType==2:#itemBased
        test.itemBased(ratingMatrix,ratingMatrix[0])
    elif recType==3:#slopeOne
        test.slopeOne(ratingMatrix,ratingMatrix[0])
    elif recType==4:#lfm
        ratingMatrix=ratingMatrix[0:10,0:20]
        # ratingMatrix=np.array([[1,3,5,2,4],[3,5,2,3,1],[5,5,4,2,1],[4,5,2,1,3],[4,4,2,3,5]])
        test.lfm_rec(ratingMatrix,5,0.001,0.01,1000)#原始评分矩阵，隐类个数，学习速率，正则化参数，迭代次数
    elif recType==5:#biaslfm
        ratingMatrix = ratingMatrix[0:10, 0:20]
        #ratingMatrix=np.array([[1,3,5,2,4],[3,5,2,3,1],[5,5,4,2,1],[4,5,2,1,3],[4,4,2,3,5]])
        test.biaslfm(ratingMatrix,5,0.001,0.01,1000)#原始评分矩阵，隐类个数，学习速率，正则化参数，迭代次数
    elif recType==6:#svdpp
        #ratingMatrix = ratingMatrix[0:10, 0:20]
        ratingMatrix=np.array([[1,3,5,2,4],[3,5,2,3,1],[5,5,4,2,1],[4,5,2,1,3],[4,4,2,3,5]])
        test.svdpp(ratingMatrix,5,0.001,0.01,1000)#原始评分矩阵，隐类个数，学习速率，正则化参数，迭代次数



