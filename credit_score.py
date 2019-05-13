# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm,skew
from scipy.special import boxcox1p
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#%%
#读取数据
train=pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\train_dataset.csv')
test=pd.read_csv(r'C:\Users\m1788\Desktop\pku\dataset\test_dataset.csv')
#%%
#用户编码特征处理(删除用户编码列)
train_Id = train['用户编码']
test_Id = test['用户编码']
train.drop('用户编码',axis=1,inplace=True)
test.drop('用户编码',axis=1,inplace=True)
#%%
#异常值处理(展示数据分布)
fig,ax=plt.subplots()
ax.scatter(train['用户账单当月总费用（元）'],train['信用分'])

plt.xlabel('cost in this month',fontsize=13)
plt.ylabel('credit score',fontsize=13)
plt.show()
#%%
#删除异常值
train = train.drop(train[train['用户账单当月总费用（元）']>800].index)
#画图
fig,ax = plt.subplots()
ax.scatter(train['用户账单当月总费用（元）'],train['信用分'])

plt.xlabel('cost in this month',fontsize=13)
plt.ylabel('credit score',fontsize=13)
plt.show()
#%%
#信用分特征处理
sns.distplot(train['信用分'],fit=norm)
#画图
(mu,sigma) = norm.fit(train['信用分'])
print('mu={:.2f},sigma={:.2f}'.format(mu,sigma))

plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('credit score distribution') 
figure=plt.figure()
stats.probplot(train['信用分'],plot=plt)
plt.show()
#%%
# 数据偏度大，用log1p函数转化，使其更加服从高斯分布。
# 最后需要将预测出的平滑数据进行还原，而还原过程就是log1p的逆运算expm1
train['信用分'] = np.log1p(train['信用分'])
#画图
sns.distplot(train['信用分'],fit=norm)
(mu,sigma) = norm.fit(train['信用分'])
print('mu={:.2f},sigma={:.2f}'.format(mu,sigma))

plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('credit score distribution') 
figure=plt.figure()
stats.probplot(train['信用分'],plot=plt)
plt.show()
#%%
#数据集连接
ntrain=train.shape[0]
ntest=test.shape[0]
y_train = train.信用分.values
all_data=pd.concat((train,test)).reset_index(drop=True)
all_data.drop(['信用分'],axis=1,inplace=True)
print('all_data size is {}'.format(all_data.shape))
#%%
#缺失数据分析
all_data.isnull().sum().head(30)
#%%
#计算缺失率
all_data_na = (all_data.isnull().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio':all_data_na})
missing_data.head(20)
#%%增加特征
#%%倾斜特征
#计算偏度
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
#skewness = pd.DataFrame({'skew':skewed_feats})
#skewness.head(28)
#%%
#对偏态分布的数据进行标准化处理(使用BOX COX)，使其更加服从正态分布
skewed_feats = skewed_feats.drop(skewed_feats[abs(skewed_feats)<1].index)
skewed_feats=skewed_feats.index
lam=0.5
num=0
for feat in skewed_feats:
    all_data[feat] = boxcox1p(all_data[feat],lam)
    num+=1
print('boxcox num is {}'.format(num))
#%%独热编码（未解决）
all_data = pd.get_dummies(all_data)
all_data.head()
#%%重新划分数据集
train=all_data[:ntrain]
test=all_data[ntrain:]
#%%定义交叉验证策略
# cross_val_score默认使用K折交叉验证策略。此处先使用KFold的shuffle参数混洗数据
# neg_mean_squared_error：负均方误差，是损失函数，优化目标是使其最小化
n_splits = 5
def nmse_cv(model):
    kf = KFold(n_splits,shuffle=True,random_state=23).get_n_splits(train.values)
    nmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring='neg_mean_squared_error',cv=kf))
    return(nmse)
#%%建立基础模型
# lasso/ElasticNet模型对异常值敏感，使用RobustScaler缩放有离群值的数据
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))
ENet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=.9,random_state=3))
GBoost=GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#%%基础模型分数
#models=[lasso,ENet,GBoost,model_xgb,model_lgb]
#names=['lasso','ELasticNet', 'GradientBoosting', 'Xgboost', 'LGBM']
#for model,name in zip(models,names):
    #score=nmse_cv(model)
    # 验证结果返回5个分数，求均值和标准差
    #print('{} score:{:.4f} ({:.4f}) \n'.format(name, score.mean(), score.std()))
#%%模型叠加(Stacking)
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
    # 将原来的模型clone出来，并且实现fit功能    
    def fit(self, X, y):
        self.clone_base_models = [list() for x in self.base_models]
        self.clone_meta_model = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # 使用K-fold的方法来进行交叉验证，将每次验证的结果作为新的特征来进行处理
        for i, model in enumerate(self.base_models):#对每个模型K折交叉验证
            for train_index, test_index in kfold.split(X, y):
                instance = clone(model)
                self.clone_base_models[i].append(instance)#???为什么要clone呢
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[test_index])
                out_of_fold_predictions[test_index, i] = y_pred
                
        # 将交叉验证预测出的结果(标签)和训练集中的标签值用元模型进行训练
        self.clone_meta_model.fit(out_of_fold_predictions, y)
        return self
        
    def predict(self, X):
        # 得到各模型预测结果平均值的二维数组
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.clone_base_models
        ])
        return self.clone_meta_model.predict(meta_features)
#%%定义评估函数
def mse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))
#%%模型训练预测评估
#叠加模型
stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost), meta_model=lasso)
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
# 前面用log1p函数转化使标签更加服从高斯分布，现用expm1将预测出的平滑数据进行还原
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
#xgb
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
#lgb
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test))
# 基于交叉验证分数给出权重
print('MSE score on train data:')
print('stacking')
print(mse(y_train, stacked_train_pred))
print('xgboost')
print(mse(y_train, xgb_train_pred))
print('LightGBM')
print(mse(y_train, lgb_train_pred))
print(mse(y_train, stacked_train_pred*0.50 + xgb_train_pred*0.10 + lgb_train_pred*0.40))
#集成预测结果
ensemble = stacked_pred*0.50 + xgb_pred*0.10 + lgb_pred*0.40
ensemble
#生成预测文件
sub = pd.DataFrame()
sub['id'] = test_Id
sub['score'] = round(ensemble).astype(int)
sub.to_csv('submit.csv', index=False)