#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import math
from com.sy.reco.similarity.cosine import Cosine

'''
这里使用正则化的线性回归方法，基于用户打分和项目特征，构成用户画像数据。
主要使用用户的特征和项目的特征
'''
class ContentRegression():
    '''
    初始化ratingMatrix,itemsFeatures,alpha, λ
    ratingMatrix:评分矩阵
    itemsFeatures:项目的特征矩阵
    alpha:学习速率
    λ:正则化参数，以防过拟合
    '''
    def __init__(self,ratingMatrix,itemsFeatures,alpha,λ):
        self.ratingMatrix=ratingMatrix
        self.itemsFeaturesMatrix=itemsFeatures
        self.alpha=alpha
        self.λ=λ

    #预测
    def predict(self,vec1,vec2):
        product=0.
        for a,b in zip(vec1,vec2):
            product+=a*b
        return product

    #迭代训练分解，max_iter:迭代次数，学习到所有的用户特征(画像)
    def iteration_train(self,max_iter):
        n_features = len(self.itemsFeaturesMatrix[0])+1  # 第0行，总共多少个特征，这里加1是线性回归中的截距项b=1
        n_users=len(self.ratingMatrix)#用户数
        n_items=len(self.ratingMatrix[0])#项目数

        #将项目特征矩阵放入items_features中，第0列为截距项1
        #其实就是为项目特征矩阵itemsFeaturesMatrix增加一列截距项全为1
        items_features=np.ones((n_items,n_features))
        items_features[:,1:]=self.itemsFeaturesMatrix
        items_features=items_features.astype(float)
        self.ratingMatrix=self.ratingMatrix.astype(float)

        #初始化用户特征矩阵，这里是需要求解的权重（在线性回归中）
        users_featuers=np.random.rand(n_users,n_features)
        users_featuers[:,0]=1.#第0列是截距项，这里设置为全1
        #迭代求解，得到所有用户的特征
        for iteration in range(max_iter):
            for user in range(n_users):
                for feature in range(n_features):
                    if feature==0:#这里是截距项，不需要正则化
                        for item in range(n_items):
                            if self.ratingMatrix[user,item]>0:#训练评过分的数据
                                #预测值与真实值的差
                                difference=self.predict(users_featuers[user],items_features[item])-self.ratingMatrix[user,item]
                                users_featuers[user,feature]-=self.alpha*difference*items_features[item,feature]#items_features[item,feature]一项代表这个项目是否有此特征，有则是1，无则是0
                    else:
                        for item in range(n_items):
                            if self.ratingMatrix[user,item]>0:#训练评过分的数据
                                difference=self.predict(users_featuers[user],items_features[item])-self.ratingMatrix[user,item]
                                users_featuers[user,feature]-=self.alpha*(difference*items_features[item,feature]+self.λ*users_featuers[user,feature])
        self.users_features=users_featuers
        self.items_features=items_features

    # 预测误差训练，convergence:误差收敛，小于这个误差，停止
    def convergence_train(self, convergence):
        n_features = len(self.itemsFeaturesMatrix[0]) + 1  # 第0行，总共多少个特征，这里加1是线性回归中的截距项b=1
        n_users = len(self.ratingMatrix)  # 用户数
        n_items = len(self.ratingMatrix[0])  # 项目数

        # 将项目特征矩阵放入items_features中，第0列为截距项1
        # 其实就是为项目特征矩阵itemsFeaturesMatrix增加一列截距项全为1
        items_features = np.ones((n_items, n_features))
        items_features[:, 1:] = self.itemsFeaturesMatrix
        items_features = items_features.astype(float)
        self.ratingMatrix = self.ratingMatrix.astype(float)

        # 初始化用户特征矩阵，这里是需要求解的权重（在线性回归中）
        users_featuers = np.random.rand(n_users, n_features)
        users_featuers[:, 0] = 1.  # 第0列是截距项，这里设置为全1
        flag=True
        # 迭代误差求解，得到所有用户的特征
        while flag:
            for user in range(n_users):
                for feature in range(n_features):
                    if feature == 0:  # 这里是截距项，不需要正则化
                        for item in range(n_items):
                            if self.ratingMatrix[user, item] > 0:  # 训练评过分的数据
                                # 预测值与真实值的差
                                difference = self.predict(users_featuers[user], items_features[item]) - \
                                             self.ratingMatrix[user, item]
                                users_featuers[user, feature] -= self.alpha * difference * items_features[
                                    item, feature]  # items_features[item,feature]一项代表这个项目是否有此特征，有则是1，无则是0
                    else:
                        for item in range(n_items):
                            if self.ratingMatrix[user, item] > 0:  # 训练评过分的数据
                                difference = self.predict(users_featuers[user], items_features[item]) - \
                                             self.ratingMatrix[user, item]
                                users_featuers[user, feature] -= self.alpha * (
                                difference * items_features[item, feature] + self.λ * users_featuers[user, feature])

            cost=0
            for user in range(n_users):
                for item in range(n_items):
                    if self.ratingMatrix[user,item]>0:
                        cost+=(1/2)*math.pow(self.ratingMatrix[user,item]-self.predict(users_featuers[user],items_features[item]),2)
                for feature in range(1,n_features):
                    cost+=(1/2)*math.pow(users_featuers[user,feature],2)
            if cost<convergence:
                break

        self.users_features = users_featuers
        self.items_features = items_features

    #user_vec是用户的评分向量，推荐给他未评分的其它项目
    #这里找到与user_vec最相似的用户对应的用户画像，然后找到user_vec未评分的项目，并利用这个用户画像与项目特征点乘，得到预测值
    def recommend(self,user_vec):
        mostSim=0.
        mostSimUser=0#最相似用户
        user_feature=np.zeros(len(self.users_features[0]))#特征数
        cosine=Cosine()#相似函数
        for user in range(len(self.ratingMatrix)):
            if np.array_equal(self.ratingMatrix[user],user_vec):#两个用户评分相同
                continue
            #计算与user_vec最相似的那个用户
            sim=cosine.similarity(self.ratingMatrix[user],user_vec)
            if sim>mostSim:
                mostSim=sim
                mostSimUser=user
        #user_feature是与user_vec最相似用户的画像特征
        user_feature=self.users_features[mostSimUser]
        #利用user_vec最相似用户的画像特征数据预测计算user_vec未评分的项目的分数
        predictRating=np.zeros(len(user_vec))
        for u_i in range(len(user_vec)):
            if user_vec[u_i]==0:
                predictRating[u_i]=self.predict(user_feature,self.items_features[u_i])
        return predictRating










































