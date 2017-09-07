#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import math
import copy

#这里使用加权的als交替最小二乘求解U,I矩阵，主要用于隐式反馈
#参考论文Collaborative Filtering for Implicit Feedback Datasets
class ALSWR():
    '''
     初始化ratingMatrix,F, λ
    ratingMatrix:评分矩阵
    F:隐因子数目
    λ:正则化参数，以防过拟合
    alpha:置信因子，用于隐式反馈评分
    '''
    def __init__(self, ratingMatrix, F, λ,alpha):
        self.ratingMatrix = ratingMatrix
        self.F = F
        self.λ = λ
        self.alpha=alpha

    #对U,I矩阵初始化，随机填充，根据经验随机数与1/sqrt(F)成正比,Pui矩阵：rui>0:1;rui=0:0，Cui矩阵是置信度权重因子：Cui=1+alpha*rui或Cui = 1+alphalog(1 + rui/ǫ).
    def __initPQ(self,userSum,itemSum):
        self.Pui=np.zeros((userSum,itemSum))#对项目有过行为：1，未有行为：0，二元
        self.Cui = np.zeros((userSum,itemSum))  # Cui矩阵是置信度权重因子：Cui=1+alpha*rui
        for user in range(userSum):
            for item in range(itemSum):
                self.Cui[user,item]=1.+self.alpha*self.ratingMatrix[user,item]#评分大则加大权重，説明用户喜欢，未评分不代表用户不喜欢，所以权重减小
                if self.ratingMatrix[user,item]>0:
                    self.Pui[user,item]=1
                else:
                    self.Pui[user,item]=0
        self.U=np.zeros((userSum,self.F))
        self.I=np.zeros((itemSum,self.F))
        for i in range(userSum):
            self.U[i]=[np.random.random()/math.sqrt(self.F) for x in range(self.F)]
        for i in range(itemSum):
            self.I[i]=[np.random.random()/math.sqrt(self.F) for x in range(self.F)]

    #使用交替二乘迭代训练分解，max_iter:迭代次数
    def iteration_train(self,max_iter):
        userSum = len(self.ratingMatrix)  # 用户个数
        itemSum = len(self.ratingMatrix[0])  # 项目个数
        self.__initPQ(userSum,itemSum)#初始化U,I矩阵
        for step in range(max_iter):
            #这时固定self.I，求解self.U
            for user in range(userSum):
                for f in range(self.F):
                    sum_1=0.
                    sum_2=0.
                    for item in range(itemSum):
                        #if self.ratingMatrix[user,item]>0:
                            eui=self.Pui[user,item]-self.predict(user,item)#误差
                            sum_1+=(eui+self.U[user,f]*self.I[item,f])*self.I[item,f]*self.Cui[user,item]
                            sum_2+=self.I[item,f]**2*self.Cui[user,item]
                    sum_2+=self.λ
                    self.U[user,f]=sum_1/sum_2
            #这里固定self.U，求解self.I
            for item in range(itemSum):
                for f in range(self.F):
                    sum_1=0.
                    sum_2=0.
                    for user in range(userSum):
                        #if self.ratingMatrix[user,item]>0:
                            eui=self.Pui[user,item]-self.predict(user,item)#误差
                            sum_1+=(eui+self.U[user,f]*self.I[item,f])*self.U[user,f]*self.Cui[user,item]
                            sum_2+=self.U[user,f]**2*self.Cui[user,item]
                    sum_2+=self.λ
                    self.I[item,f]=sum_1/sum_2
        return np.round(np.dot(self.U, self.I.T), 0)  # 返回全部，两个矩阵相乘

    #预测打分，用户的行与项目的列
    def predict(self,user,item):
        I_T=self.I.T#项目矩阵转置
        pui=np.dot(self.U[user,:],I_T[:,item])
        return pui















