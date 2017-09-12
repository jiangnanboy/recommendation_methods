#!/usr/bin/Python
# -*- coding: utf-8 -*-
__author__="yan.shi"

import numpy as np
from com.sy.reco.similarity.cosine import Cosine

'''
这里结合user_based与contentbased方法推荐
将用户评过分的项目的所有特征构成一个用户偏好向量，扩展用户的评分矩阵，在评分的后面加入对项目的特征偏好列项
基于user的相似计算，并预测评分（原来的user_based相似度只依靠评分计算，现在是评分和特征偏好共同参与计算相似度，这里的特征
偏好可以看作是基于内容的）
'''
class ContentUserCF():

    '''
    ratingMatrix：评分矩阵
    itemsFeatures：项目特征矩阵
    itemsNames：项目名
    '''
    def __init__(self,ratingMatrix,itemsFeatures):
        self.ratingMatrix=ratingMatrix
        self.itemsFeatures=itemsFeatures

    #每个用户的项目的特征分布，得到每个用户的特征喜好分布
    def getUser_Items_Features(self,user_rating):
        #用户特征
        user_features=np.zeros(self.items_n_features)
        #每个特征计数
        features_count=np.zeros(self.items_n_features)
        for item in range(self.n_items):
            if user_rating[item]>0:
                #每个项目中的特征*该用户对这个项目的评分
                user_features+=self.itemsFeatures[item]*user_rating[item]
                #所有用户评过分的项目的特征的和
                features_count+=self.itemsFeatures[item]
        #求特征的平均值
        for fea in range(self.items_n_features):
            if features_count[fea]>0:
                user_features[fea]/=float(features_count[fea])
        return user_features

    #得到用户的评分，及在项目上的特征偏好矩阵user_features_matrix，格式是行为用户，列为评分及特征
    def getUsers_Ratings_Features(self):
        #一个item有多少个特征
        self.items_n_features=len(self.itemsFeatures[0])
        #总共多少个项目
        self.n_items=len(self.ratingMatrix[0])
        #总共多少用户
        self.n_users=len(self.ratingMatrix)
        #新建和扩展用户特征，这里是用户对每个项目的评分以及项目特征
        self.user_features_matrix=np.zeros((self.n_users,self.items_n_features+self.n_items))
        #每个用户评分的均值,reshape(-1,1)表示将数组转为1列
        users_mean=np.array([self.ratingMatrix[i][self.ratingMatrix[i]>0].mean() for i in range(self.n_users)]).reshape(-1,1)
        '''
        #所有用户评分减去各自的评分均值
        users_diff=[]
        for u in self.n_users:
            user_rating=[]
            for i in self.n_items:
                if self.ratingMatrix[u][i]>0:
                    rating=self.ratingMatrix[u][i]-users_mean[u]#评分减去评分均值
                else:
                    rating=0.
                user_rating.append(rating)
            users_diff.append(user_rating)
        users_diff=np.array(users_diff)
        '''
        #评分矩阵放入user_features_matrix中
        self.user_features_matrix[:,:self.n_items]=self.ratingMatrix
        #计算用户在项目上的特征偏好向量
        for u in range(self.n_users):
            user_vec=self.ratingMatrix[u]#用户的评分向量
            #从第n_items列开始，放入每个用户的特征喜好分布
            self.user_features_matrix[u,self.n_items:]=self.getUser_Items_Features(user_vec)

    #返回K个相似用户
    def _KNeighbours(self,item,simMatrix,K):
        neighbours=[]
        count=0
        for user in range(self.n_users):
            if simMatrix[user,item]>0:
                neighbours.append(simMatrix[user])
                count+=1
            if count==K:
                break
        return np.array(neighbours)

    #预测评分1，公式：Pu,j=Uv+sum[sim(u,v)(Vj-Vv)]/|sum(sim(u,v)|
    #Uv是用户U的评分均值，sim(u,v)是用户v与用户u的相似度，Vj是用户V对项目j的评分，Vv是用户V的评分均值
    def _rating(self,user_vec,item,neighbours):
        score = 0.0
        simSum = 0.0
        for i in range(len(neighbours)):
            score+=neighbours[i,-1]*float(neighbours[i,item]-neighbours[i][neighbours[i]>0][:-1].mean())
            simSum+=abs(neighbours[i,-1])
        if simSum>0:
            score=np.round(user_vec[user_vec>0].mean()+score/simSum,0)
        else:  # 没有近邻，返回自己的评分均值
            score = np.round(self.user_vec[self.user_vec > 0].mean(), 0)
        if score > 5:
            return 5.0
        elif score < 1:
            return 1.0
        return score

    #预测评分，用户评分向量user_vec，相似用户阈值
    def predictRating(self,user_vec,K):
        cos=Cosine()#相似类
        #self.n_users用户数
        #列数，包括项目+特征
        n_features=self.n_items+self.items_n_features
        #用于计算相似的数据矩阵，最后一列用于存入与目标用户的相似度
        similarity_matrix=np.zeros((self.n_users,n_features+1))
        similarity_matrix[:,:-1]=self.user_features_matrix
        user_fea_vec=np.append(user_vec,self.getUser_Items_Features(user_vec))#获取用户的所有特征
        #计算所有用户与目标用户user_fea_vec的相似度
        for user in range(self.n_users):
            if np.array_equal(similarity_matrix[user,:-1],user_fea_vec):#去掉与自己的计算相似度
                similarity_matrix[user,n_features]=0.
            else:
                #计算相似度
                similarity_matrix[user,n_features]=cos.similarity(similarity_matrix[user,:-1],user_fea_vec)

        #对相似度排序
        similarity_matrix=similarity_matrix[similarity_matrix[:,n_features].argsort()][::-1]
        user_predict=np.zeros(len(user_vec))#存入预测评分
        #K个相似用户，评分
        for item in range(self.n_items):
            if user_vec[item]==0:#未评过分的项目
                #K个最相似的对item评过分的用户
                neighbours=self._KNeighbours(item,similarity_matrix,K)
                #对item评分
                user_predict[item]=self._rating(user_vec,item,neighbours)
        return user_predict
