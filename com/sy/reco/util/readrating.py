#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
import pandas as pd

#读取要计算的数据
class ReadRating():

    #读取rating数据
    def readRatingData(self,ratingPath):
        df_rating=pd.read_csv(ratingPath)
        return df_rating

    #读取item内容特征数据
    def readItemContent(self,itemContentPath):
        itemContentMatrix=pd.read_csv(itemContentPath)
        return itemContentMatrix

     #读取item数据
    def readItemData(self,itemPath):
        df_item=pd.read_csv(itemPath)
        return df_item

     #读取user数据
    def readUserData(self,userPath):
        df_user=pd.read_csv(userPath)
        return df_user

    #获取user的评分均值
    def average_userRating(self, userRating):
        userRating = userRating.astype(float)
        for i in range(len(userRating)):
            userRating[i][userRating[i] == 0] = sum(userRating[i]) / float(len(userRating[i][userRating[i] > 0]))
        return userRating

    #获取item的评分均值
    def average_itemRating(self, itemRating):
        itemRating = itemRating.astype(float)
        for i in range(len(itemRating[0])):
            itemRating[:, i][itemRating[:, i] == 0] = sum(itemRating[:, i]) / float(
                len(itemRating[:, i][itemRating[:, i] > 0]))
        return itemRating

if __name__=='__main__':
    read=ReadRating()
    df_rating=read.readRatingData('G:\\python workspace\\recommendation_methods\\data\\ratingmatrix.csv')
    #print(df_rating.head(4))
    #print(df_rating.values[:,1:])#所有评分
    print(len(df_rating))