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
        item_info,df_write=self.readData(ratingPath,itemPath,savetoPath)
        self.readItemContent(item_info,df_write,savetoPath)

    #处理原始数据，将评分数据存储
    def readData(self,ratingPath,itemPath,savetoPath):
        print('开始处理数据...')
        #评分数据
        df_rating=pd.read_csv(ratingPath,sep='\t',header=None)
        #项目列表
        item_info=pd.read_csv(itemPath,sep='|',header=None)
        #拿到item数据的第一列，项目名
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
            df_write.loc[i]=[i]+tmpItem #一行数据，用户Id，及其对item的评分

        removieItem=list(set(removieItem))
        df_write.drop(df_write.columns[removieItem],axis=1,inplace=True) #inplace=True 直接替换原数组
        #df_write.to_csv(savetoPath,index=None)#处理后的数据存入文件中
        return item_info,df_write #item_info存储的是关于item的内容属性,df_write存储的是经预处理的评分数据

    #读取和存储关于item内容
    def readItemContent(self,item_info,df_write,savetoPath):
        #获得项目名对应的数字
        itemList=[int(itemNum.split(';')[-1]) for itemNum in df_write.columns[1:] ]#获得所有列名对应的数字,df_write中的第0列是人名，从第1列开始是项目名
        #这是项目从属的类型，比如未知，动作，冒险等
        itemKind=['unknown','Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary',
              'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
              'Romance','Sci-Fi','Thriller','War','Western']
        item_kind=pd.DataFrame(columns=[['item_id']+itemKind])
        start=5
        count=0
        for itemNum in itemList:#列名对应的数字
            item_kind.loc[count]=[itemNum]+item_info.iloc[itemNum-1][start:].tolist()
            count+=1
        item_kind.to_csv(savetoPath,index=None)

if __name__=='__main__':
    datapre=DataProcess('G:\\python workspace\\recommendation_methods\\data\\ratings.csv',
                        'G:\\python workspace\\recommendation_methods\\data\\movies.csv',
                        'G:\\python workspace\\recommendation_methods\\data\\moviesContent.csv')











































































