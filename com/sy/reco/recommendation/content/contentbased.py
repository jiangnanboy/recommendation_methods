#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
from com.sy.reco.similarity.cosine import Cosine
'''
本方法是基于内容的推荐，目的是生成用户的画像，利用用户评过分的项目的特征分布作一个总结，获得用户在所有特征上的喜好
分布向量，比如一部电影，其特征可能是动作，科幻等。缺点是对一些无法描述的项目不好推荐。优点是独立于用户评分情况，
缓解冷启动问题。
这里使用所有用户喜好特征的电影的平均分作为用户画像，根据这个向量计算与推荐与其最相似的电影
'''
class ContentBased():
    #itemsNmaesList是所有项目的名称向量，itemsFeatures是所有项目的特征分布，user_vec是用户评过的项目向量
    def __init__(self,itemsNamesList,itemsFeatures,user_vec):
        self.itemsNamesList=itemsNamesList
        self.itemsFeatures=itemsFeatures
        self.user_vec=user_vec

    #计算用户画像，推荐与画像最相似的top_k个项目
    def getRecItems(self,top_k):
        n_features=len(self.itemsFeatures[0])#第0行，多少个特征
        n_items=len(self.user_vec)#此用户的item数（也就是所有item数）
        user_mean=self.user_vec[self.user_vec>0.].mean()#获得该用户评分的均值
        diff_uvec=self.user_vec-user_mean#用户的评分减去该用户的平均评分
        user_features=np.zeros(n_features).astype(float)#这里是用户画像，是用户对所有特征的偏好程度
        item_featuers_sum=np.zeros(n_features)#这里是暂存包含某个特征的item数量
        for item in range(n_items):
            if self.user_vec[item]>0:#用户评过分的项目
                for feature in range(n_features):
                    user_features[feature]+=diff_uvec[item]*self.itemsFeatures[item,feature]
                    item_featuers_sum[feature]+=self.itemsFeatures[item,feature]#若item包含这个特征，就加1，否则加0
        #特征平均
        for feature in range(n_features):
            if item_featuers_sum[feature]>0:
                user_features[feature]=user_features[feature]/float(item_featuers_sum[feature])

        # 得到了用户画像user_features（对所有特征的偏好程度），就可以用它计算与用户未评过分的项目的相似度
        self._calSimilarity(user_features,n_items,top_k)

    #计算用户画像与所有其未评过分的项目的相似度
    def _calSimilarity(self,user_features,n_items,top_k):
        itemSimlarity = np.zeros(n_items)
        cosSim = Cosine()  # 相似度
        for item in range(n_items):
            if self.user_vec[item] == 0:  # 用户未评过分的项目
                itemSimlarity[item] = cosSim.similarity(user_features, self.itemsFeatures[item])
        # 对相似度排序，输出最高相似度的项目，降序排序
        order_item_sim = np.argsort(itemSimlarity)[::-1][0:top_k]  # 这个返回的是索引，前top_k个
        for index in order_item_sim:
            print('%d,%s,%f'% (index,self.itemsNamesList[index],itemSimlarity[index]))





