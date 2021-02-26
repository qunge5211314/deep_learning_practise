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


def load_data():
    pass
