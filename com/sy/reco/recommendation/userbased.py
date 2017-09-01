#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import pandas as pd
from com.sy.reco import similarity

#基于相似user推荐
class UserBased():

    #ratingMatrix是所有用户的评分矩阵，
    # user_vec是需要与评分矩阵的计算的当前用户评分向量
    def __init__(self,ratingMatrix,user_vec):
        self.ratingMatrix=ratingMatrix
        self.user_vec=user_vec

    #返回K个最相似的用户
    def kNeighbours(self,K):
        ratingMatrix=self.ratingMatrix.astype(float)
        self.nrows=len(ratingMatrix)#行数，多少个用户
        self.ncols=len(ratingMatrix[0])#列数，多少个item
        kSimData=np.zeros((self.nrows,self.ncols+1))#存放K个相似的数据，最后一列是存放与目标用户的相似度
        kSimData[::-1]=ratingMatrix#除最后一列外，将评分数据放入kSimData中
        #计算与目标用户评分向量user_vec最相似用户
        for user in range(self.nrows):
            if np.array_equal(kSimData[user,:-1],self.user_vec):#评分相同
                kSimData[user,self.ncols]=0.0#最后一列存放相似度
                continue
            kSimData[user,self.ncols]=similarity.cosine.similarity(kSimData[user,:-1],self.user_vec)
        #降序排列，截取前k个
        k_simNeighbours=kSimData[kSimData[:,self.ncols].argsort()][::-1][0:K,:]
        return k_simNeighbours

    #返回预测计算的评分,k_simNeighbours为k个最相似的用户的数据
    def getRratingPredict(self,k_simNeighbours):
        #存放用户未评过分的预测值
        userPredictRating=np.zeros(len(self.user_vec))
        for itemNum in range(self.ncols):
            if self.user_vec[itemNum]==0:
                simNeighbours=[]
                for userNum in range(len(k_simNeighbours)):
                    if k_simNeighbours[userNum,itemNum]>0:
                        simNeighbours.append(k_simNeighbours[userNum])#存储该用户评分不为0的一行记录
                neighbours=np.array(simNeighbours)
                userPredictRating[itemNum]=self.ratingPredicting_1(neighbours,itemNum)
        return userPredictRating

    #预测评分1，公式：Pu,j=Uv+sum[sim(u,v)(Vj-Vv)]/|sum(sim(u,v)|
    #Uv是用户U的评分均值，sim(u,v)是用户v与用户u的相似度，Vj是用户V对项目j的评分，Vv是用户V的评分均值
    def ratingPredicting_1(self,neighbours,itemNum):
        score=0
        simSum=0
        for i in range(len(neighbours)):
            meanRating=float(neighbours[i][itemNum]-neighbours[i][neighbours[i]>0][:-1].mean())#用户i的评分均值
            score+=neighbours[i][-1]*meanRating
            simSum+=abs(neighbours[i][-1])
        if simSum>0:
            score=np.round(self.user_vec[self.user_vec>0].mean()+(score/simSum),0)
        else:#没有近邻，返回自己的评分均值
            score=np.round(self.user_vec[self.user_vec>0].mean(),0)
        if score>5:
            return 5.0
        elif score<1:
            return 1.0
        return score

    #预测评分2，公式：Pu,j=Uv+Qu*sum[sim(u,v)(Vj-Vv)/Qv]/|sum(sim(u,v)|,Uv是用户U的评分均值
    # sim(u,v)是用户v与用户u的相似度，Vj是用户V对项目j的评分，Vv是用户V的评分均值，Qu是用户u的评分标准差，Qv用户v的评分标准差
    def ratingPredicting_2(self,neighbours,itemNum):
        score = 0
        simSum = 0
        for i in range(len(neighbours)):
            meanRating = float(neighbours[i][itemNum] - neighbours[i][neighbours[i] > 0][:-1].mean())  # 用户i的评分均值,考虑0评分是用户未评分的，所以去除
            score += neighbours[i][-1] * meanRating/np.std(neighbours[i][neighbours[i]>0][:-1])#除以标准差
            simSum += abs(neighbours[i][-1])
        if simSum > 0:
            score = np.round(self.user_vec[self.user_vec > 0].mean() + np.std(self.user_vec[self.user_vec>0])*score / simSum, 0)#目标用户的评分标准差
        else:  # 没有近邻，返回自己的评分均值
            score = np.round(self.user_vec[self.user_vec > 0].mean(), 0)
        if score > 5:
            return 5.0
        elif score < 1:
            return 1.0
        return score
























