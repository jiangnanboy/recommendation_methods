#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import math

#这里使用非负矩阵求解U,I矩阵，使用交替最小二乘迭代计算，要求元素非负
#这里正则化使用的是岭回归和套索回归的综合，alpha是正则化选择参数
class NMF():
    '''
     初始化ratingMatrix,F, λ
    ratingMatrix:评分矩阵
    F:隐因子数目
    λ:正则化参数，以防过拟合
    alpha:正则化选择参数，默认正则化是岭回归，即平方正则化
    '''
    def __init__(self, ratingMatrix, F, λ,alpha=0):
        self.ratingMatrix = ratingMatrix
        self.F = F
        self.λ = λ
        self.alpha=alpha

    #对U,I矩阵初始化，随机填充非零（0，1），根据经验随机数与1/sqrt(F)成正比
    def __initPQ(self,userSum,itemSum):
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
                        if self.ratingMatrix[user,item]>0:
                            eui=self.ratingMatrix[user,item]-self.predict(user,item)#误差
                            sum_1+=(eui+self.U[user,f]*self.I[item,f])*self.I[item,f]
                            sum_2+=self.I[item,f]**2
                    sum_1-=self.alpha*self.λ
                    sum_2+=(1-self.alpha)*self.λ
                    self.U[user,f]=sum_1/sum_2
            #这里固定self.U，求解self.I
            for item in range(itemSum):
                for f in range(self.F):
                    sum_1=0.
                    sum_2=0.
                    for user in range(userSum):
                        if self.ratingMatrix[user,item]>0:
                            eui=self.ratingMatrix[user,item]-self.predict(user,item)#误差
                            sum_1+=(eui+self.U[user,f]*self.I[item,f])*self.U[user,f]
                            sum_2+=self.U[user,f]**2
                    sum_1-=self.alpha*self.λ
                    sum_2+=(1-self.alpha)*self.λ
                    self.I[item,f]=sum_1/sum_2
        return np.round(np.dot(self.U, self.I.T), 0)  # 返回全部，两个矩阵相乘

    #预测打分，用户的行与项目的列
    def predict(self,user,item):
        I_T=self.I.T#项目矩阵转置
        pui=np.dot(self.U[user,:],I_T[:,item])
        return pui
