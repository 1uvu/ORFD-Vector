#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tests-vec.py
@Desc    :
@Project :   orfd-platform
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/05/29 0:49        the freer      2.1
'''
import pandas as pd
import numpy as np
from sklearn.externals import joblib

from main import main
from Core.classific import cv, my_test
from setting import VEC_TRAIN_PATH, VEC_TEST_PATH, VEC_LARGE_PATH

def vec():
	data = pd.read_csv(VEC_TRAIN_PATH)
	print("训练集规模：", data.shape)
	print("训练集标签分布：\n", pd.value_counts(data.get("62")))
	test = pd.read_csv(VEC_TEST_PATH)
	print("测试集规模：", test.shape)
	print("测试集标签分布：\n", pd.value_counts(test.get("62")))
	target = "62"
	# 去掉ID和属性列
	x_columns = [x for x in data.columns if x not in [target]]
	t_columns = [x for x in test.columns if x not in [target]]
	data_x = np.array(data[x_columns])
	data_y = np.array(data[target])
	test_x = np.array(test[t_columns])
	test_y = np.array(test[target])
	clf = joblib.load("core/models/rf_62_vec.m")
	print("平衡数据集随机森林模型交叉验证评估结果：")
	cv(data_x, data_y, clf)
	print("-----------------------------------")
	
	print("平衡数据集随机森林模型测试集评估结果：")
	my_test(test_x, test_y, clf)
	print("-----------------------------------")
	
	large = pd.read_csv(VEC_LARGE_PATH)
	print("完全不平衡数据集规模：", large.shape)
	print("完全不平衡数据集标签分布：\n", pd.value_counts(large.get("62")))
	# 去掉ID和属性列
	l_columns = [x for x in large.columns if x not in [target]]
	large_x = np.array(large[l_columns])
	large_y = np.array(large[target])
	my_test(large_x, large_y, clf)


def predict():
	with open("./input_test.txt", "rb") as f:
		input_data = [data.decode().strip() for data in f.readlines()]
	input_data.append("")
	data = input_data
	rf = joblib.load("./core/models/rf_62_vec.m")
	vec_proba, desc = main(data, "rf")
	print("特征向量概率向量：", vec_proba)
	if vec_proba[0] < 0.6:
		print("\n ----> 预判为假！！")
	else:
		print("\n ----> 预判为真！！")

if __name__ == '__main__':
	print("开始测试特征向量分类模型性能：")
	vec()
	print("开始测试对输入数据识别：")
	predict()