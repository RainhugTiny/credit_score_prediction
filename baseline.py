# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:29:58 2019

@author: m1788
"""
#%%模块导入
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',100)#显示最大列数
#%%数据导入
train_data = pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\train_dataset.csv')
test_data = pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\test_dataset.csv')
submit_data = pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\submit_example.csv')
#%%数据分割及处理
valid_data = train_data[40000:]
train_data = train_data[:40000]
valid_data.drop('用户编码',axis=1,inplace=True)
train_data.drop('用户编码',axis=1,inplace=True)
test_data.drop('用户编码',axis=1,inplace=True)
#%%baseline,mae:14.9256
def lgb_eval(train_df,valid_df):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    #取出训练集和验证集的标签
    X_train = train_df.drop('信用分',axis=1)
    Y_train = train_df.loc[:,'信用分'].values
    X_valid = valid_df.drop('信用分',axis=1)
    Y_valid = valid_df.loc[:,'信用分'].values 
    #训练模型
    #learing_rate=0.04-14.8420,0.06-14.8132
    reg_model = lgb.LGBMRegressor(max_depth=5,n_estimators=500,num_threads=4,
                                  learning_rate=0.06,objective='mse')
    reg_model.fit(X_train,Y_train)
    #预测模型
    y_pred=reg_model.predict(X_valid)
    #residual = y_pred-Y_valid
    print(mean_absolute_error(Y_valid,y_pred),end=' ')
    #特征重要性
    print('')
    #feature=X_train.columns
    #fe_importance=reg_model.feature_importances_
    #print(pd.DataFrame({'fe':feature,'importance':fe_importance}).sort_values(by='importance',ascending=False))
    #残差绘图
    #plt.clf()
    #plt.figure(figsize=(15,4))
    #plt.plot([Y_train.min(),Y_train.max()],[0,0],color='red')
    #plt.scatter(x=y_pred,y=residual)
    #plt.show()
    return y_pred
#%%
#eval_df=lgb_eval(train_data,valid_data)
#print(eval_df)
#%%原生特征丢弃尝试,丢弃'当月火车类应用使用次数'，14.9146，'当月物流快递类应用使用次数':14.9205,'当月是否体育场馆消费':14.9190
#for col in train_data.columns:
#    if col != '信用分':
#        print('drop col:{}'.format(col))
#        tmp_train_data = train_data.drop([col],axis=1)
#        tmp_valid_data = valid_data.drop([col],axis=1)
#        eval_df = lgb_eval(tmp_train_data,tmp_valid_data)
#全部丢弃:14.9066
train_data = train_data.drop(['当月火车类应用使用次数','当月物流快递类应用使用次数','当月是否体育场馆消费'],axis=1)
valid_data = valid_data.drop(['当月火车类应用使用次数','当月物流快递类应用使用次数','当月是否体育场馆消费'],axis=1)
#eval_data = lgb_eval(train_data,valid_data)
#%%特征工程
#tmp_train_data = train_data.copy()
#tmp_valid_data = valid_data.copy()
def add_features(dataset):
    #充值方式（若充值金额不为整则为微信支付宝支付（有红包），充值金额为整数则为非微信支付宝支付）14.9694
    #dataset['rechargeMethod'] = 0
    #dataset['rechargeMethod'][(dataset['缴费用户最近一次缴费金额（元）']-round(dataset['缴费用户最近一次缴费金额（元）']
    #)) != 0] = 1
    #月平均充值金额14.9327
    #dataset['averageRecharge'] = dataset['缴费用户最近一次缴费金额（元）'] / (dataset['用户最近一次缴费距今时长（月）']+1)
    #平均每月余额14.9327
    #dataset['平均每月余额'] = dataset['缴费用户最近一次缴费金额（元）'] / (dataset['用户最近一次缴费距今时长（月）'] + 1)
    #- dataset['用户近6个月平均消费值（元）']
    #用户账单稳定性14.8956(可用2-1)
    dataset['stability'] = dataset['用户账单当月总费用（元）'] / (dataset['用户近6个月平均消费值（元）'])
    #账单余额比14.9287
    #dataset['cost_left_rate'] = dataset['用户账单当月总费用（元）'] / dataset['用户当月账户余额（元）']
    #缴费金额是否能覆盖当月账单14.97
    #dataset['缴费金额是否能覆盖当月账单']=dataset['缴费用户最近一次缴费金额（元）']-dataset['用户账单当月总费用（元）']
    #是否去过高档商场14.9366
    #dataset['是否去过高档商场']=dataset['当月是否到过福州山姆会员店']+dataset['当月是否逛过福州仓山万达']
    #信用不良客户(降低，one-hot后降低更多)14.9193
    #dataset['bad_user'] = dataset['用户实名制是否通过核实']+dataset['是否黑名单客户']+dataset['缴费用户当前是否欠费缴费']+dataset['是否4G不健康客户']
    #用户网龄等级(无提升)
    def map_netAge(x):
        if x<=49:
            return 0
        elif x<=94:
            return 1
        elif x<=139:
            return 2
        else:
            return 3
    #dataset['netAge_level'] = dataset['用户网龄（月）'].map(lambda x:map_netAge(x))  
    #用户年龄分级（无提升）
    def map_Age(x):
        if x<=30:
            return 0
        elif x<36:
            return 1
        elif x<45:
            return 2
        else:
            return 3
    #dataset['Age_level'] = dataset['用户年龄'].map(lambda x:map_Age(x))    
    #零一变量的组合
    #1-14.9311
    #dataset['是否大学生_黑名单']=dataset['是否大学生客户']+dataset['是否黑名单客户']
    #2-14.9286
    #dataset['是否去过高档商场']=dataset['当月是否到过福州山姆会员店']+dataset['当月是否逛过福州仓山万达']
    #dataset['是否去过高档商场']=dataset['是否去过高档商场'].map(lambda x:1 if x>=1 else 0)
    #3-14.9229
    #dataset['是否_商场_电影']=dataset['是否去过高档商场']*dataset['当月是否看电影']
    #4-14.9520
    #dataset['是否_商场_旅游']=dataset['是否去过高档商场']*dataset['当月是否景点游览']
    #5-14.8926(可用2-2)
    dataset['是否去过高档商场']=dataset['当月是否到过福州山姆会员店']+dataset['当月是否逛过福州仓山万达']
    dataset['是否去过高档商场']=dataset['是否去过高档商场'].map(lambda x:1 if x>=1 else 0)
    dataset['是否_电影_旅游']=dataset['当月是否看电影']*dataset['当月是否景点游览']
    #6-14.9393
    #dataset['是否_商场_电影_旅游']=dataset['是否去过高档商场']*dataset['当月是否看电影']*dataset['当月是否景点游览']
    #7-14.9304
    #dataset['6个月平均占比总费用']=dataset['用户近6个月平均消费值（元）']/(dataset['用户账单当月总费用（元）']+1)
    #增加可用的两个特征:14.8842
    return dataset
#tmp_train_data = add_features(tmp_train_data)
#tmp_valid_data = add_features(tmp_valid_data)
train_data = add_features(train_data)
valid_data = add_features(valid_data)
eval_data=lgb_eval(train_data,valid_data)
#%%
train_data.describe()

