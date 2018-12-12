# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:21:13 2018

@author: lj
"""
import numpy as np
import math
import copy

def load_data(file_name):
    '''导入数据
    input:  file_name(string):文件的存储位置
    output: feature_data(mat):特征
            label_data(mat):标签
            n_class(int):类别的个数
    '''
    # 1、获取特征
    f = open(file_name)  # 打开文件
    feature_data = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label.append(int(lines[-1]))      
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    
    return np.mat(feature_data), label

def Normalization(data):
    '''样本数据归一化
    input:data(mat):样本特征矩阵
    output:Nor_feature(mat):归一化的样本特征矩阵
    '''
    m,n = np.shape(data)
    Nor_feature = copy.deepcopy(data) 
    sample_sum = np.sqrt(np.sum(np.square(data),axis = 1))   
    for i in range(n):
        Nor_feature[:,i] = Nor_feature[:,i] / sample_sum
        
    return Nor_feature

def distance(X,Y):
    '''计算两个样本之间的距离
    '''
    return np.sum(np.square(X-Y),axis = 1)

def distance_mat(Nor_trainX,Nor_testX):
    '''计算待测试样本与所有训练样本的欧式距离
    input:Nor_trainX(mat):归一化的训练样本
          Nor_testX(mat):归一化的测试样本
    output:Euclidean_D(mat):测试样本与训练样本的距离矩阵
    '''
    m,n = np.shape(Nor_trainX)
    p = np.shape(Nor_testX)[0]
    Euclidean_D = np.mat(np.zeros((p,m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i,j] = distance(Nor_testX[i,:],Nor_trainX[j,:])[0,0]
    return Euclidean_D

def Gauss(Euclidean_D,sigma):
    '''测试样本与训练样本的距离矩阵对应的Gauss矩阵
    input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
          sigma(float):Gauss函数的标准差
    output:Gauss(mat):Gauss矩阵
    '''
    m,n = np.shape(Euclidean_D)
    Gauss = np.mat(np.zeros((m,n)))
    for i in range(m):
        for j in range(n):
            Gauss[i,j] = math.exp(- Euclidean_D[i,j] / (2 * (sigma ** 2)))
    return Gauss

def Prob_mat(Gauss_mat,labelX):
    '''测试样本属于各类的概率和矩阵
    input:Gauss_mat(mat):Gauss矩阵
          labelX(list):训练样本的标签矩阵
    output:Prob_mat(mat):测试样本属于各类的概率矩阵
           label_class(list):类别种类列表
    '''
    ## 找出所有的标签类别
    label_class = []
    for i in range(len(labelX)):
        if labelX[i] not in label_class:
            label_class.append(labelX[i])
    
    n_class = len(label_class)
    ## 求概率和矩阵
    p,m = np.shape(Gauss_mat)
    Prob = np.mat(np.zeros((p,n_class)))
    for i in range(p):
        for j in range(m):
            for s in range(n_class):
                if labelX[j] == label_class[s]:
                    Prob[i,s] += Gauss_mat[i,j]
    Prob_mat = copy.deepcopy(Prob)
    Prob_mat = Prob_mat / np.sum(Prob,axis = 1)
    return Prob_mat,label_class

def calss_results(Prob,label_class):
    '''分类结果
    input:Prob(mat):测试样本属于各类的概率矩阵
          label_class(list):类别种类列表
    output:results(list):测试样本分类结果
    '''
    arg_prob = np.argmax(Prob,axis = 1) ##类别指针
    results = []
    for i in range(len(arg_prob)):
        results.append(label_class[arg_prob[i,0]])
    return results
                    

if __name__ == '__main__':
    # 1、导入数据
    print ("--------- 1.load data ------------")
    trainX, labelX = load_data("data.txt")
    # 2、样本数据归一化
    Nor_trainX = Normalization(trainX)
    Nor_testX = Normalization(trainX[100:300,:]) 
    # 3、计算Gauss矩阵
    Euclidean_D = distance_mat(Nor_trainX,Nor_testX)
    Gauss_mat = Gauss(Euclidean_D,0.1)
    Prob,label_class = Prob_mat(Gauss_mat,labelX)
    # 4、求测试样本的分类
    predict_results = calss_results(Prob,label_class)
    
    
    
    
    
    
    
    
    
    
    
    
