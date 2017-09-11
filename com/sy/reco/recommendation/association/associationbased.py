#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np

'''
使用关联规则的推荐方法，利用支持度和置信度构建规则X->Y，这里X和Y分别只包含一个元素
支持度：support(X->Y)=包含X,Y的事务数/总事务数
置信度：confidence(X->Y)=support(X->Y)/support(X)
首先计算1-项集的支持度，过滤小于min_support的项，在1-项集中构建2-项集，并计算支持度和过滤，然后计算置信度，过滤小于min_confidence的项
计算所有的item-item的置信度，构建置信度矩阵。最后利用用户的评分向量与这个置信度矩阵相乘。得到推荐列表，这里不是评分预测
（这里还可以利用提升度lift(X->Y）来检验是否强关联规则，lift(X->Y)=confidence(X->Y)/support(Y)，lift>1为强关联，lift<=1为
无效的关联规则
'''
class AssociationBased():

    '''
    ratingMatrix：评分矩阵
    itemNameList：项目名称
    min_support：最小支持度
    min_confidence：最小置信度
    itemScoreThreshold：项目最小评分
    '''
    def __init__(self,ratingMatrix,itemNameList,min_support=0.1,min_confidence=0.1,itemScoreThreshold=3):
        self.ratingMatrix=ratingMatrix
        self.itemNameList=itemNameList
        self.min_support=min_support
        self.min_confidence=min_confidence
        self.itemScoreThreshold=itemScoreThreshold

    #返回最小支持度的项目集，以及所有项目的支持度,support(X)=X在多少条事务中/总事务数
    def calSupport(self,transactions_set,items_frozenset,n_trans):
        itemsCount={}#每个item出现的次数(一个用户算一个事务，在所有事务出现的总和)
        for item_sets in transactions_set:
            for item in items_frozenset:
                if item.issubset(item_sets):#item是否是item_sets子集
                    itemsCount.setdefault(item,0)
                    itemsCount[item]+=1
        #n_trans=len(transactions_set)#总事务数
        freq_sets=[]#频繁集
        sets_support={}#项目的支持度
        for key,value in itemsCount.items():
            support =value/float(n_trans)
            if support>=self.min_support:
                freq_sets.insert(0,key)
            sets_support[key]=support
        return freq_sets,sets_support

    #构建2-项集
    def cal2FreqItems(self,list_freqsets_init,setlen):
        freq_2_items=[]
        n_trans=len(list_freqsets_init)
        for i in range(n_trans):
            setlist1=set(list_freqsets_init[i])
            for j in range(i+1,n_trans):
                setlist2=set(list_freqsets_init[j])
                freq_2_items.append(setlist1.union(setlist2))
        return freq_2_items

    #计算关联规则X->Y，得到所有item-item置信度矩阵
    def calRules(self):
        n_items=len(self.ratingMatrix[0])#项目个数
        transactions=[]#事务集
        #获得每个用户评分大于阈值的数据
        for user in self.ratingMatrix:
            #获得所有评分大于itemScoreThreshold的项目的索引
            ratingIndex=[i for i in range(n_items) if user[i]>self.itemScoreThreshold]
            if len(ratingIndex)>0:
                transactions.append(ratingIndex)
        n_trans=len(transactions)#总事务数
        #展开，将所有用户评过分的且大于阈值的项目index放入flatItems列表中，包括重复的
        flatItems=[]
        for itemIndexList in transactions:
            for itemIndex in itemIndexList:
                flatItems.append(itemIndex)
        #frozenset，将所有flatItems中的所有项转为frozenset类型，去掉了重复元素
        items_frozenset=map(frozenset,[[item] for item in frozenset(flatItems)])
        #将每个用户事务数据transactions中的每行list转为每行set，第行为一个用户的set元素
        transactions_set=map(set,transactions)
        #下面是计算大于最小支持度的频繁项集freqsets_init，以及每单个项目的支持度set_sets_support
        list_1_freqsets_init,self.set_sets_support=self.calSupport(transactions_set,items_frozenset,n_trans)
        #构建2-项集
        items_tmp=self.cal2FreqItems(list_1_freqsets_init,2)
        #计算大于最小支持度的2-项集，及所有2-项集的支持度
        self.freq_2_sets,set_2_sets_support=self.calSupport(transactions_set,items_tmp,n_trans)
        self.set_sets_support.update(set_2_sets_support)#将1-项集和2-项集更新在一起，里面放的是项目及其支持度
        self.confidence_matrix=np.zeros((n_items,n_items))#项目-项目的置信度矩阵
        #迭人2-频繁项集，计算两个项目的置信度
        for freqset in self.freq_2_sets:
            list_items=[]
            for item in freqset:
                list_items.append(frozenset([item]))
            self.calConfidenceMatrix(freqset,list_items)

    #confidence(X-Y)置信度为:support(X,Y)/support(X)
    def calConfidenceMatrix(self,freqset,list_freqlist):
        for item in list_freqlist:
            confidence=self.set_sets_support[freqset]/self.set_sets_support[item]
            if confidence>=self.min_confidence:
                self.confidence_matrix[item][freqset-item]=confidence

    #得到所有的item-item置信度，可以根据这个矩阵乘以用户的评分向量，这里不是评分测试，而是一个推荐列表
    def getRecItems(self,user_vec):
        rec_vec=np.dot(user_vec,self.confidence_matrix)
        #未评过分的项目
        nowatch=[index for index in range(len(user_vec)) if user_vec[index]==0]
        nowatchitems=np.array(self.itemNameList)[nowatch]
        sortedIndex=np.argsort(rec_vec)#排序，返回的是index
        recitems=np.array(self.itemNameList)[sortedIndex]
        recitems=[item for item in recitems if item in nowatchitems]
        return recitems[::-1]
