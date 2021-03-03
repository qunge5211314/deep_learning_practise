#!python3
# -*- coding: utf-8 -*-
# Author: JustinHan
# Date: 2021-02-26
# Introduce: 第二章节任务1
# MLP快速搭建非线性二分类模型task：
# 基于task1_data数据，建立mlp模型，实现非线性边界二分类。
#
# 1、数据分离:test_size=0.2, random_state=0；
# 2、建模并训练模型（迭代1000次），计算训练集、测试集准确率；
# 3、可视化预测结果
# 4、继续迭代6000次，重复步骤2-3
# 5、迭代1-10000次（500为间隔），查看迭代过程中的变化(可视化结果、准确率）
# 模型结构：一层隐藏层，25个神经元，激活函数：sigmoid
# Dependence
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score


# 加载原始数据
def load_raw_data(path):
    raw_data = pd.read_csv(path)
    print("raw data:\n", raw_data.describe())
    return raw_data


# 数据清洗
def clean_data(raw_data):
    # 1.缺失值处理
    cleaned_data1 = raw_data.dropna()
    # 2.去重
    cleaned_data = cleaned_data1.drop_duplicates()
    print("cleaned data:\n", cleaned_data.describe())
    return cleaned_data


def show_cleaned_data(cleaned_data):
    x = cleaned_data.drop(['y'], axis=1)
    y = cleaned_data.loc[:, 'y']
    fig1 = plt.figure(figsize=(5, 5))
    label1 = plt.scatter(x.loc[:, 'x1'][y == 1], x.loc[:, 'x2'][y == 1])
    label0 = plt.scatter(x.loc[:, 'x1'][y == 0], x.loc[:, 'x2'][y == 0])
    plt.legend((label1, label0), ('label1', 'label0'))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('raw data')
    plt.show()


if __name__ == '__main__':
    path = "task1_data.csv"
    # 获取原始数据
    raw_data = load_raw_data(path)
    # 数据清洗
    cleaned_data = clean_data(raw_data)
    # 数据可视化
    # show_cleaned_data(cleaned_data)
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(cleaned_data.drop(['y'], axis=1), cleaned_data.loc[:, "y"])
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 建立mlp模型
    mlp_model = Sequential()
    mlp_model.add(Dense(units=25, input_dim=2, activation='sigmoid'))
    mlp_model.add(Dense(units=1, activation='sigmoid'))
    print(mlp_model.summary())
    # 模型求解参数配置
    mlp_model.compile(optimizer='adam', loss='binary_crossentropy')
    # 模型训练
    mlp_model.fit(x_train, y_train, epochs=6000)
    # 训练数据结果预测
    y_train_predict = mlp_model.predict_classes(x_train)
    # 表现评估
    accuracy_train = accuracy_score(y_train, y_train_predict)
    print(accuracy_train)
    y_test_predict = mlp_model.predict_classes(x_test)
    accuracy_test = accuracy_score(y_test, y_test_predict)
    print(accuracy_test)
