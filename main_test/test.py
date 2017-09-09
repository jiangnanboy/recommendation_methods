#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
from com.sy.reco.recommendation.cf.itembased import ItemBased
from com.sy.reco.recommendation.cf.userbased import UserBased
from com.sy.reco.recommendation.cf.slopeone import SlopeOne
from com.sy.reco.recommendation.matrix_factorization.als import ALS
from com.sy.reco.recommendation.matrix_factorization.als_wr import ALSWR
from com.sy.reco.recommendation.matrix_factorization.biaslfm import BiasLFM
from com.sy.reco.recommendation.matrix_factorization.lfm import LFM
from com.sy.reco.recommendation.matrix_factorization.nmf import NMF
from com.sy.reco.recommendation.matrix_factorization.svdpp import SVDPP
from com.sy.reco.recommendation.content.contentbased import ContentBased
from com.sy.reco.recommendation.content.contentregression import ContentRegression
from com.sy.reco.similarity.cosine import Cosine
from com.sy.reco.util.dataprocess import DataProcess
from com.sy.reco.util.readrating import ReadRating

class Test():
    #数据预处理
    def dataProcess(self,ratingPath,itemPath,savetoPath):
        DataProcess(ratingPath,itemPath,savetoPath)
    #读取评分数据
    def readRatingMatrix(self,ratingMatrixPath):
        read=ReadRating()
        ratingMatrix=read.readRatingData(ratingMatrixPath)
        return ratingMatrix.values[:,1:]#去除第一列是用户的标号，返回的是numpy中的ndarray类型,返回评分数据
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
    #ALS
    def als(self,ratingMatrix,F,λ,max_iter):
        alsRec=ALS(ratingMatrix,F,λ)
        predictRating=alsRec.iteration_train(max_iter)
        print(predictRating)
    #ALSWR
    def alsWR(self,ratingMatrix,F,λ,α,max_iter):
        als_wr=ALSWR(ratingMatrix,F,λ,α)
        predictRating=als_wr.iteration_train(max_iter)
        print(predictRating)
    #NMF
    def nmf(self,ratingMatrix,F,λ,α,max_iter):
        nmf_rec=NMF(ratingMatrix,F,λ,α)
        predictRating=nmf_rec.iteration_train(max_iter)
        print(predictRating)
    #contentbased
    def contentRec(self,itemsNamesList,itemsFeatures,user_vec):
        content_rec=ContentBased(itemsNamesList,itemsFeatures,user_vec)
        content_rec.getRecItems(10)
    #contentbased2
    def contentRegre(self,ratingMatrix,itemsFeatures,user_vec,alpha,λ,max_iter):
        gression=ContentRegression(ratingMatrix,itemsFeatures,alpha,λ)
        gression.iteration_train(max_iter)
        predictRating=gression.recommend(user_vec)
        print(predictRating)

if __name__=='__main__':
    test=Test()
    #所有用户的评分矩阵数据，行为用户，列为项目
    ratingMatrix = test.readRatingMatrix('G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv')
    recType=11
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
    elif recType==7:#ALS
        # ratingMatrix = ratingMatrix[0:10, 0:20]
        #ratingMatrix = np.array([[1, 3, 5, 2, 4], [3, 5, 2, 3, 1], [5, 5, 4, 2, 1], [4, 5, 2, 1, 3], [4, 4, 2, 3, 5]])
        ratingMatrix = np.array([[1, 3, 0, 2, 0], [3, 0, 2, 0, 1], [5, 5, 4, 0, 1], [0, 5, 2, 0, 3], [4, 0, 2, 3, 5]])
        test.als(ratingMatrix,2,0.01,1000)#原始评分矩阵，隐类个数，正则化参数，迭代次数
    elif recType==8:#ALSWR
        # ratingMatrix = ratingMatrix[0:10, 0:20]
        ratingMatrix = np.array([[1, 3, 0, 2, 0], [3, 0, 2,0, 1], [5, 5, 4, 0, 1], [0, 5, 2, 0, 3], [4, 0, 2, 3, 5]])
        test.alsWR(ratingMatrix,2,0.01,0.5,500)#原始评分矩阵，隐类个数，正则化参数，置信因子，迭代次数
    elif recType==9:#NMF
        ratingMatrix = np.array([[1, 3, 0, 2, 0], [3, 0, 2, 0, 1], [5, 5, 4, 0, 1], [0, 5, 2, 0, 3], [4, 0, 2, 3, 5]])
        test.nmf(ratingMatrix, 2, 0.01, 0, 1000)  # 原始评分矩阵，隐类个数，正则化参数，正则化参数选择因子，迭代次数
    elif recType==10:#contentbased
        #评分矩阵
        ratingMatrixPath='G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv'
        read = ReadRating()
        rating_Matrix = read.readRatingData(ratingMatrixPath)
        #item内容特征矩阵
        itemContentMatrixPath='G:\\python workspace\\recommendation_methods\\data\\moviesContent.csv'
        itemContentMatrix=read.readItemContent(itemContentMatrixPath)
        #item名
        itemsNamesList=list(rating_Matrix.columns[1:])#ratingMatrix第0列是用户列，从1列开始是item列
        #所有item的特征矩阵
        itemsFeatures=itemContentMatrix.values[:,1:]#itemContentMatrix第0列是item名列，其余是item的特征分布
        test.contentRec(itemsNamesList,itemsFeatures,ratingMatrix[0])
    elif recType==11:#contentgression
        read=ReadRating()
        # item内容特征矩阵
        itemContentMatrixPath = 'G:\\python workspace\\recommendation_methods\\data\\moviesContent.csv'
        itemContentMatrix = read.readItemContent(itemContentMatrixPath)
        # 所有item的特征矩阵
        itemsFeatures = itemContentMatrix.values[:, 1:]  # itemContentMatrix第0列是item名列，其余是item的特征分布
        test.contentRegre(ratingMatrix,itemsFeatures,ratingMatrix[0],0.001,0.01,100)#评分数据矩阵，项目特征矩阵，测试的第0行用户，学习速率，正则化参数，迭代次数


