#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import pandas as pd
import copy
import collections
from scipy import linalg
import math
from collections import defaultdict

'''
使用pandas将评分数据转为DataFrame评分矩阵，并存储为评分矩阵格式
'''
class DataProcess():

    def __init__(self,ratingPath,itemPath,savetoPath):
        self.readData(ratingPath,itemPath,savetoPath)

    '''
    处理原始数据
    '''
    def readData(self,ratingPath,itemPath,savetoPath):
        print('开始处理数据...')
        df_rating=pd.read_csv(ratingPath,sep='\t',header=None)
        item_info=pd.read_csv(itemPath,sep='|',header=None)
        #拿到item数据的第一列
        itemList=[item_info[1].tolist()[index]+';'+str(index+1) for index in range(len(item_info[1].tolist()))]
        #item数据大小（多少个item）
        itemSize=len(itemList)
        #user数据大小（多少个user）
        userSize=len(df_rating[0].drop_duplicates().tolist())
        #去除少于50个用户评分的item
        min_ratings=50
        #计算评分数据中的每个item被评分的次数
        item_counts=collections.Counter(list(df_rating[1]))
        #增加user列
        df_write=pd.DataFrame(columns=['user']+itemList)
        #需要删除的item
        removieItem=[]
        for i in range(1,userSize):
            tmpItem=[0 for j in range(itemSize)]
            #获取user id==i的用户的数据
            df_tmp=df_rating[df_rating[0]==i]
            for k in df_tmp.index:
                #获取大于50个用户评分数的item
                if item_counts[df_tmp.ix[k][1]]>=min_ratings:
                    tmpItem[df_tmp.ix[k][1]-1]=df_tmp.ix[k][2] #tmpItem存的是评分，下标是item序列-1
                else:
                    removieItem.append(df_tmp.ix[k][1])#需要删除的item对应的数字序列
            df_write.loc[i]=[i]+tmpItem #一行数据，用户Id，及item评分

        removieItem=list(set(removieItem))
        df_write.drop(df_write.columns[removieItem],axis=1,inplace=True) #inplace=True 直接替换原数组
        df_write.to_csv(savetoPath,index=None)#处理后的数据存入文件中

    '''
    获取user的评分均值
    '''
    def average_userRating(self,userRating):
        userRating=userRating.astype(float)
        for i in range(len(userRating)):
            userRating[i][userRating[i]==0]=sum(userRating[i])/float(len(userRating[i][userRating[i]>0]))
        return userRating

    '''
    获取item的评分均值
    '''
    def average_itemRating(self,itemRating):
        itemRating=itemRating.astype(float)
        for i in range(len(itemRating[0])):
            itemRating[:,i][itemRating[:,i]==0]=sum(itemRating[:,i])/float(len(itemRating[:,i][itemRating[:,i]>0]))
        return itemRating












































































