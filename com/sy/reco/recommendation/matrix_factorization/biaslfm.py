#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import math

'''
BiasLFM(bias latent factor model)带偏置项的隐语义推荐模型，加入三个偏置项（所有评分的平均u，用户偏置项bu表示用户的评分习惯和物品没关系，
物品偏置项bi表示物品接受的评分中和用户没关系的因素）矩阵分解,训练得到U,I矩阵,以及用户偏置项和物品偏置项
对user-item评分矩阵进行分解为U、I矩阵，再利用随机梯度下降（函数值下降最快的方向）迭代求解出U,I矩阵以及bu和bi，最后用U*I预测得出user对item的预测评分
这里U矩阵是user对每个隐因子的偏好程度，I矩阵是item在每个隐因子中的分布
'''
class BiasLFM():
    '''
    初始化ratingMatrix,F, alpha, λ
    ratingMatrix:评分矩阵
    F:隐因子数目
    alpha:学习速率
    λ:正则化参数，以防过拟合
    '''
    def __init__(self, ratingMatrix, F, alpha, λ):
        self.ratingMatrix = ratingMatrix
        self.F = F
        self.alpha = alpha
        self.λ = λ

    #对U,I矩阵初始化i，随机填充，根据经验随机数与1/sqrt(F)成正比,bu向量与bi向量初始化为全0,u是所有有评分的全局平均
    def __initPQ(self,userSum,itemSum):
        self.U=np.zeros((userSum,self.F))
        self.I=np.zeros((itemSum,self.F))
        self.bu=np.zeros(userSum)
        self.bi=np.zeros(itemSum)
        self.u=np.mean(self.ratingMatrix[self.ratingMatrix>0])#全局均值
        for i in range(userSum):
            self.U[i]=[np.random.random()/math.sqrt(self.F) for x in range(self.F)]
        for i in range(itemSum):
            self.I[i]=[np.random.random()/math.sqrt(self.F) for x in range(self.F)]

    #预测打分，用户的行与项目的列
    def predict(self,user,item):
        I_T=self.I.T#项目矩阵转置
        pui=self.u+self.bu[user]+self.bi[item]
        pui+=np.dot(self.U[user,:],I_T[:,item])
        return pui

    # 迭代训练分解，max_iter:迭代次数
    def iteration_train(self, max_iter):
        userSum=len(self.ratingMatrix)#用户个数
        itemSum=len(self.ratingMatrix[0])#项目个数
        self.__initPQ(userSum,itemSum)#初始化U,I,bu,bi
        for step in range(max_iter):
            for user in range(userSum):
                for item in range(itemSum):
                    if self.ratingMatrix[user, item] > 0:  # 未评分的项目不参与计算
                        eui=self.ratingMatrix[user,item]-self.predict(user,item)#真实值减去预测的值
                        self.bu[user]+=self.alpha*(eui-self.λ*self.bu[user])#更新bu
                        self.bi[item]+=self.alpha*(eui-self.λ*self.bi[item])#更新bi
                        for f in range(self.F):
                            self.U[user,f]+=self.alpha*(self.I[item,f]*eui-self.λ*self.U[user,f])#更新U
                            self.I[item,f]+=self.alpha*(self.U[user,f]*eui-self.λ*self.I[item,f])#更新I

        predictRating=[]
        for user in range(userSum):
            userItemRating=[]
            for item in range(itemSum):
                pui=self.predict(user,item)
                userItemRating.append(pui)
            predictRating.append(userItemRating)
        return np.round(np.array(predictRating),0)

    # 预测误差训练，convergence:误差收敛，小于这个误差,则终止训练
    def convergence_train(self, convergence):
        userSum = len(self.ratingMatrix)  # 用户个数
        itemSum = len(self.ratingMatrix[0])  # 项目个数
        self.__initPQ(userSum, itemSum)  # 初始化U,I,bu,bi
        flag = True
        while flag:
            for user in range(userSum):
                for item in range(itemSum):
                    if self.ratingMatrix[user, item] > 0:  # 未评分的项目不参与计算
                        eui = self.ratingMatrix[user, item] - self.predict(user, item)  # 真实值减去预测的值
                        self.bu[user] += self.alpha * (eui - self.λ * self.bu[user])  # 更新bu
                        self.bi[item] += self.alpha * (eui - self.λ * self.bi[item])  # 更新bi
                        for f in range(self.F):
                            self.U[user, f] += self.alpha * (self.I[item, f] * eui - self.λ * self.U[user, f])  # 更新U
                            self.I[item, f] += self.alpha * (self.U[user, f] * eui - self.λ * self.I[item, f])  # 更新I

            cost = 0  # 训练误差
            for user in range(userSum):
                for item in range(itemSum):
                    if self.ratingMatrix[user, item] > 0:
                        cost += (1/2)*math.pow(self.ratingMatrix[user, item] - self.predict(user,item), 2)
                        cost+=(1/2)*self.λ*(math.pow(self.bu[user],2)+math.pow(self.bi[item],2))
                        for f in range(self.F):
                            cost += (1 / 2) * self.λ * (math.pow(self.U[user, f], 2) + math.pow(self.I[item, f], 2))
            if cost < convergence:
                flag = False

        predictRating = []
        for user in range(userSum):
            userItemRating = []
            for item in range(itemSum):
                pui = self.predict(user, item)
                userItemRating.append(pui)
            predictRating.append(userItemRating)
        return np.round(np.array(predictRating), 0)









