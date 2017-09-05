#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import math

'''
LFM(latent factor model)隐语义推荐模型，利用矩阵分解来拟合原始评分矩阵,训练得到U,I矩阵
对user-item评分矩阵进行分解为U、I矩阵，再利用随机梯度下降（函数值下降最快的方向）迭代求解出U,I矩阵，最后用U*I预测得出user对item的预测评分
这里U矩阵是user对每个隐因子的偏好程度，I矩阵是item在每个隐因子中的分布
最小化误差平方函数，加入正则化是为了减少过拟合
'''
class LFM():
    '''
    初始化ratingMatrix,F, α, λ
    ratingMatrix:评分矩阵
    F:隐因子数目
    α:学习速率
    λ:正则化参数，以防过拟合
    '''
    def __init__(self,ratingMatrix,F,α,λ):
        self.ratingMatrix=ratingMatrix
        self.F=F
        self.α=α
        self.λ=λ

    #对U,I矩阵初始化i，随机填充，根据经验随机数与1/sqrt(F)成正比
    def __initPQ(self,userSum,itemSum):
        U=np.zeros((userSum,self.F))
        I=np.zeros((itemSum,self.F))
        for i in range(userSum):
            U[i]=[np.random.random()/math.sqrt(self.F) for x in range(self.F)]
        for i in range(itemSum):
            I[i]=[np.random.random()/math.sqrt(self.F) for x in range(self.F)]
        return  U,I

    #预测打分，用户的行与项目的列
    def predict(self,user,item,U,I):
        I_T=I.T#项目矩阵转置
        pui=np.dot(U[user,:],I_T[:,item])
        return pui

    #迭代训练分解，max_iter:迭代次数
    def iteration_train(self,max_iter):
        userSum = len(self.ratingMatrix)  # 用户个数
        itemSum = len(self.ratingMatrix[0])  # 项目个数
        U,I=self.__initPQ(userSum,itemSum)
        for step in range(max_iter):
            for user in range(userSum):
                for item in range(itemSum):
                    if self.ratingMatrix[user,item]>0:#未评分的项目不参与计算
                        eui=self.ratingMatrix[user,item]-self.predict(user,item,U,I)#真实值减去预测的值
                        for f in range(self.F):
                            U[user,f]+=self.α*(I[item,f]*eui-self.λ*U[user,f])#更新
                            I[item,f]+=self.α*(U[user,f]*eui-self.λ*I[item,f])#更新
            self.α*=0.9#对学习参数进行衰减，使用算法尽快收敛
        return np.dot(U,I.T)#返回全部，两个矩阵相乘

    #预测误差训练，convergence:误差收敛，小于这个误差
    def convergence_train(self,convergence):
        userSum = len(self.ratingMatrix)  # 用户个数
        itemSum = len(self.ratingMatrix[0])  # 项目个数
        U,I = self.__initPQ(userSum, itemSum)
        flag=True
        while flag:
            for user in range(userSum):
                for item in range(itemSum):
                    if self.ratingMatrix[user,item]>0:#未评分的项目不参与计算
                        eui=self.ratingMatrix[user,item]-self.predict(user,item,U,I)#真实值减去预测的值
                        for f in range(self.F):
                            U[user,f]+=self.α*(I[item,f]*eui-self.λ*U[user,f])#更新
                            I[item,f]+=self.α*(U[user,f]*eui-self.λ*I[item,f])#更新
            self.α*=0.9#对学习参数进行衰减，使用算法尽快收敛

            cost=0#误差
            for user in range(userSum):
                for item in range(itemSum):
                    if self.ratingMatrix[user,item]>0:
                        cost+=math.pow(self.ratingMatrix[user,item]-np.dot(U[user],I.T[:,item]),2)
                        for f in range(self.F):
                            cost+=(1/2)*self.λ*(math.pow(U[user,f],2)+math.pow(I[item,f],2))
            if cost<convergence:
                flag=False
        return np.dot(U,I.T)#返回全部，两个矩阵相乘






















