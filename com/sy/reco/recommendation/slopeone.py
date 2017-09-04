#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np

'''
基于用户对项目评分偏差的slopeone推荐
'''
class SlopeOne():
    def __init__(self,ratingMatrix,user_vec):
        self.ratingMatrix=ratingMatrix
        self.user_vec=user_vec

    #计算所有item的偏差矩阵，公式：dev(i,j)=sum(ui-uj)/user(i,j)
    #两两项目评分的偏差除以两个项目被共同评分的用户数
    def itemDeviationMatrix(self):
        itemSize=len(self.ratingMatrix[0])#项目数
        self.itemDevMatrix=np.zeros((itemSize,itemSize))#两个项目的偏差矩阵
        self.itemCommonMatrix=np.zeros((itemSize,itemSize))#两个项目被共同评分的用户数
        for i in range(itemSize):
            for j in range(itemSize):
                if j>=i:#三角阵
                    self.itemDevMatrix[i,j],self.itemCommonMatrix[i,j]=self._deviation(self.ratingMatrix[:,i],self.ratingMatrix[:,j],itemSize)
                else:
                    self.itemDevMatrix[i,j]=-self.itemDevMatrix[j,i]
                    self.itemCommonMatrix[i,j]=self.itemCommonMatrix[j,i]

    #返回两个向量评分的偏差和共同评分用户数
    def _deviation(self,vec1,vec2,itemSize):
        common=0
        vec1Sum=0
        vec2Sum=0
        for i in range(itemSize):
            if vec1[i]==0 or vec2[i]==0:
                continue
            common+=1
            vec1Sum+=vec1[i]
            vec2Sum+=vec2[i]
        if common==0:#没有共同评分
            return 1000.0,0
        return float(vec1Sum-vec2Sum)/common,common

    #返回最小偏差的前k个项目
    def _getMiniDevKItem(self,index,K):
        devation=np.argsort(self.itemDevMatrix[index])#偏差从小到大排序
        devation=devation[devation!=index]#除去本身
        count=0
        itemDevation=[]
        for index in devation:
            if self.user_vec[index]>0:
                itemDevation.append(index)
                count+=1
            if count==K:
                break
        return itemDevation

    #返回预测计算的评分公式:Pu,j=sum((dev(j,i)+ui)*common(j,i))/sum(common(j,i)
    #dev(j,i)是项目j对项目i的评分偏差,ui是用户u对项目i的评分，common(j,i)是共同被多少用户评过分
    def _ratingPredicting(self,index,itemDevation):
        score=0.0
        commonSum=0.0
        for i in itemDevation:
            if abs(self.itemDevMatrix[index,i])!=1000:
                score+=(self.itemDevMatrix[index,i]+self.user_vec[i])*self.itemCommonMatrix[index,i]
                commonSum+=self.itemCommonMatrix[index,i]
        if commonSum==0:
            return 0.
        score=np.round(score/commonSum,0)
        if score>5:
            return 5
        elif score<1:
            return 1
        return score

    #返回该用户全部未评分项目的预测得分，K是某个项目最小偏差的前K个项目
    def getRatingPredict(self,K):
        itemSize = len(self.user_vec)
        userPredictRating = np.zeros(itemSize)
        for i in range(itemSize):
            if self.user_vec[i]==0:#如果用户未评分
                itemDevation=self._getMiniDevKItem(i,K)
                userPredictRating[i]=self._ratingPredicting(i,itemDevation)
        return userPredictRating
























