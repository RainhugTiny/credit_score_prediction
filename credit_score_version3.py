# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:09:14 2019

@author: m1788
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import skew
from scipy.special import boxcox1p
pd.set_option('display.max_columns',100)#显示最大列数
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['font.family'] = ['sans-serif']
#%%数据载入
train_data = pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\train_dataset.csv')
test_data = pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\test_dataset.csv')
submit_data = pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\submit_example.csv')
#%%数据信息：数据类型和数据分布
print(train_data.info())
print(train_data.describe())
#%%异常值处理
#报错
#train_data = train_data.drop(train_data[train_data['用户年龄']==0].index)
#train_data = train_data.drop(train_data[train_data['用户账单当月总费用（元）']>800].index)
#正常(降低)
#train_data.loc[train_data['用户年龄']==0,'用户年龄']=train_data['用户年龄'].mode()
#处理离散点(降低)
def base_process(data):
    transform_value_feature=['用户年龄','用户网龄（月）','当月通话交往圈人数','近三个月月均商场出现次数','当月网购类应用使用次数','当月物流快递类应用使用次数'
                            ,'当月金融理财类应用使用总次数','当月视频播放类应用使用次数','当月飞机类应用使用次数','当月火车类应用使用次数','当月旅游资讯类应用使用次数']
    user_fea=['缴费用户最近一次缴费金额（元）','用户近6个月平均消费值（元）','用户账单当月总费用（元）','用户当月账户余额（元）']
    log_features=['当月网购类应用使用次数','当月金融理财类应用使用总次数','当月物流快递类应用使用次数','当月视频播放类应用使用次数']
    
    #处理离散点
    for col in transform_value_feature+user_fea+log_features:
        #取出最高99.9%值
        ulimit=np.percentile(train_data[col].values,99.9)
        #取出最低0.1%值
        llimit=np.percentile(train_data[col].values,0.1)
        data.loc[train_data[col]>ulimit,col]=ulimit
        data.loc[train_data[col]<llimit,col]=llimit
        
    #for col in user_fea+log_features:
        #data[col]=data[col].map(lambda x:np.log1p(x))
    
    return data

#train_data=base_process(train_data)
#test_data=base_process(test_data)
#print(train_data.dtypes.head(30))
#%%删除特征
train_data = train_data.drop(['当月火车类应用使用次数','当月物流快递类应用使用次数','当月是否体育场馆消费'],axis=1)
test_data = test_data.drop(['当月火车类应用使用次数','当月物流快递类应用使用次数','当月是否体育场馆消费'],axis=1)
#%%特征工程
#增加特征
def add_features(dataset):
    #充值方式（若充值金额不为整则为微信支付宝支付（有红包），充值金额为整数则为非微信支付宝支付）
    #dataset['rechargeMethod'] = 0
    #dataset['rechargeMethod'][(dataset['缴费用户最近一次缴费金额（元）']-round(dataset['缴费用户最近一次缴费金额（元）']
    #)) != 0] = 1
    #月平均充值金额
    #dataset['averageRecharge'] = dataset['缴费用户最近一次缴费金额（元）'] / (dataset['用户最近一次缴费距今时长（月）']+1)
    #平均每月余额(无提升)
    #dataset['平均每月余额'] = dataset['缴费用户最近一次缴费金额（元）'] / (dataset['用户最近一次缴费距今时长（月）'] + 1)
    #- dataset['用户近6个月平均消费值（元）']
    #用户账单稳定性
    dataset['stability'] = dataset['用户账单当月总费用（元）'] / (dataset['用户近6个月平均消费值（元）'])
    #账单余额比
    #dataset['cost_left_rate'] = dataset['用户账单当月总费用（元）'] / dataset['用户当月账户余额（元）']
    #缴费金额是否能覆盖当月账单
    #dataset['缴费金额是否能覆盖当月账单']=dataset['缴费用户最近一次缴费金额（元）']-dataset['用户账单当月总费用（元）']
    #是否去过高档商场
    #dataset['是否去过高档商场']=dataset['当月是否到过福州山姆会员店']+dataset['当月是否逛过福州仓山万达']
    #信用不良客户(降低，one-hot后降低更多)
    #dataset['bad_user'] = dataset['用户实名制是否通过核实']+dataset['是否黑名单客户']+dataset['缴费用户当前是否欠费缴费']+dataset['是否4G不健康客户']
    #一个组合
    dataset['是否去过高档商场']=dataset['当月是否到过福州山姆会员店']+dataset['当月是否逛过福州仓山万达']
    dataset['是否去过高档商场']=dataset['是否去过高档商场'].map(lambda x:1 if x>=1 else 0)
    dataset['是否_电影_旅游']=dataset['当月是否看电影']*dataset['当月是否景点游览']
    #用户网龄等级
    def map_netAge(x):
        if x<=48:
            return 0
        elif x<=94:
            return 1
        elif x<=139:
            return 2
        else:
            return 3
    #dataset['netAge_level'] = dataset['用户网龄（月）'].map(lambda x:map_netAge(x))
    return dataset
train_data = add_features(train_data)
test_data = add_features(test_data)
#%%删除用户编码
test_id = test_data['用户编码']
train_data.drop('用户编码',axis=1,inplace=True)
test_data.drop('用户编码',axis=1,inplace=True)
#%%
#print(train_data.dtypes[train_data.dtypes != 'object'].index)
#%%倾斜特征
#计算偏度
#numeric_feats = train_data.dtypes[train_data.dtypes != 'object'].index
#skewed_feats = train_data[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
#skewness = pd.DataFrame({'skew':skewed_feats})
#print(skewness.head(40))
#%%
#对偏态分布的数据进行标准化处理(使用BOX COX)，使其更加服从正态分布(有效)
#skewed_feats = skewed_feats.drop(skewed_feats[abs(skewed_feats)<5].index)
#skewed_feats=skewed_feats.index
#lam=0.5
#num=0
#for feat in skewed_feats:
#    train_data[feat] = boxcox1p(train_data[feat],lam)
#    test_data[feat] = boxcox1p(test_data[feat],lam)
#    num+=1
#print('boxcox num is {}'.format(num))
#%%one-hot编码
#train_data = pd.get_dummies(train_data,columns=['用户话费敏感度','bad_user'])
#test_data = pd.get_dummies(test_data,columns=['用户话费敏感度','bad_user'])
#%%训练
#参数
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': 5,
    'lambda_l2': 5, 'lambda_l1': 0,'nthread': 8
}
params2 = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': 5,
    'lambda_l2': 5, 'lambda_l1': 0,'nthread': 8,
    'seed': 89
}


cv_pred_all = 0
train_pred_all = 0
en_amount = 3
for seed in range(en_amount):
    #K折交叉验证
    NFOLDS = 5
    train_label = train_data['信用分']
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    kf = kfold.split(train_data, train_label)
    #训练使用的数据
    train_data_use = train_data.drop(['信用分','是否黑名单客户'], axis=1)
    test_data_use = test_data.drop(['是否黑名单客户'], axis=1)
    #初始化测试集和训练集预测结果
    cv_pred = np.zeros(test_data.shape[0])
    train_pred = np.zeros(train_data.shape[0])
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], \
        train_label[train_fold], train_label[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1,early_stopping_rounds=50)
        #测试集
        cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
        #训练集
        train_pred += bst.predict(train_data_use, num_iteration=bst.best_iteration)
    train_pred /= NFOLDS
    cv_pred /= NFOLDS
    train_pred_all += train_pred
    cv_pred_all += cv_pred
train_pred_all /= en_amount
cv_pred_all /= en_amount


cv_pred_all2 = 0
train_pred_all2 = 0
en_amount = 3
for seed in range(en_amount):
    #K折交叉验证
    NFOLDS = 5
    train_label = train_data['信用分']
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=(seed + 2019))
    kf = kfold.split(train_data, train_label)
    #训练使用的数据
    train_data_use = train_data.drop(['信用分','是否黑名单客户'], axis=1)
    test_data_use = test_data.drop(['是否黑名单客户'], axis=1)
     #初始化测试集和训练集预测结果
    cv_pred = np.zeros(test_data.shape[0])
    train_pred = np.zeros(train_data.shape[0])
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], \
        train_label[train_fold], train_label[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params2, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1,early_stopping_rounds=50)
        cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
        train_pred += bst.predict(train_data_use, num_iteration=bst.best_iteration)
    train_pred /= NFOLDS
    cv_pred /= NFOLDS
    train_pred_all2 += train_pred
    cv_pred_all2 += cv_pred
train_pred_all2 /= en_amount
cv_pred_all2 /= en_amount
#%%定义评估函数
def mse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))
def mae(y,y_pred):
    return mean_absolute_error(y,y_pred)
#%%评价指标MAE
print('MAE score on train data:')
train_label=train_data.信用分.values
print(mae(train_pred_all,train_label))
print(mae(train_pred_all2,train_label))
print(mae(train_pred_all*0.5 + train_pred_all2*0.5,train_label))
#生成结果文件
sub = pd.DataFrame()
sub['id'] = test_id
sub['score'] = cv_pred_all*0.5+ cv_pred_all2*0.5
sub['score']=(round(sub['score'])).astype(int)
sub.to_csv('submit.csv', index=False)
#结果统计baseline
#MAE score on train data:
#13.623855143948475
#13.781554434373733
#13.680232163564467
#skewed 5
#MAE score on train data:
#13.618954567285426
#13.814311874776742
#13.69418950731934