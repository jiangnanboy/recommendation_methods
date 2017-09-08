#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
'''
基于项目的协同推荐
'''
class ItemBased():
    def __init__(self,ratingMatrix,user_vec):
        self.ratingMatrix=ratingMatrix#评分矩阵
        self.user_vec=user_vec#用户的评分向量

    #计算item相似矩阵(两两计算),simType为使用的相似计算方法
    def itemSimilarityMatrix(self,simType):
        itemSize=len(self.ratingMatrix[0])#多少个item
        self.itemSimMatrix=np.zeros((itemSize,itemSize))#item相似矩阵初始化
        for iIndex in range(itemSize):
            for jIndex in range(itemSize):
                if jIndex>=iIndex:#只计算三角矩阵
                    self.itemSimMatrix[iIndex,jIndex]=simType.similarity(self.ratingMatrix[:,iIndex],self.ratingMatrix[:,jIndex])
                else:
                    self.itemSimMatrix[iIndex,jIndex]=self.itemSimMatrix[jIndex,iIndex]
                print("%d,%d,%f" %(iIndex,jIndex,self.itemSimMatrix[iIndex,jIndex]))
    #返回k个最相似的项目评分
    def _getMostSimKItem(self,index,K):
        simItems=np.argsort(self.itemSimMatrix[index])[::-1][0:K+1]#最相似前K个item(这里取k+1是为了下面去除它自身与自身的相似度),返回的是索引
        simItems=simItems[simItems!=index]#去除自身与自身的相似度
        similarityItems=[]
        for i in simItems:
            if self.user_vec[i]>0:
                similarityItems.append(i)
        return similarityItems

    # 返回预测计算的评分公式：Pu,j=sum[sim(j,i)(Ui)]/|sum(sim(j,i)|
    # Ui是用户U对i的评分，sim(j,i)是项目j与项目i的相似度
    def _ratingPredicting(self,index,simItems):
        score=0.0
        simSum=0.0
        for i in simItems:
            score+=self.itemSimMatrix[index,i]*self.user_vec[i]
            simSum+=abs(self.itemSimMatrix[index,i])
        if simSum>0:
            score=np.round(score/simSum,0)
        else:#项目没有近邻，返回该项目的评分均值
            score=np.round(self.ratingMatrix[:,index][self.ratingMatrix[:,index]>0].mean(),0)
        if score>5:
            return 5.0
        elif score<1:
            return 1.0
        return score

    #返回该用户全部未评分项目的预测得分，K是某个项目最相似的前K个最相似项目
    def getRatingPredict(self,K):
        itemSize=len(self.user_vec)
        userPredictRating=np.zeros(itemSize)
        for index in range(itemSize):
            if self.user_vec[index]==0:#如果用户未评分
                simItems=self._getMostSimKItem(index,K)
                userPredictRating[index]=self._ratingPredicting(index,simItems)
        return userPredictRating


