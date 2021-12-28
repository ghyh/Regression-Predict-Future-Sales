 This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# read files into csv files
training_data = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test_data = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

training_data.describe()

# Identify outliers
# Ref: https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9,5))
plt.xlim(-50,2500)
# Figure for item_cnt_day
plt.boxplot(x=training_data.item_cnt_day,flierprops={'marker':'x','markersize':6},vert=False)

# Figure for item_price
plt.figure(figsize=(9,5))
plt.xlim(training_data.item_price.min(),training_data.item_price.max()*1.5)
plt.boxplot(x=training_data.item_price,flierprops={'marker':'x','markersize':6},vert=False)

# Excluding outliers
training_data = training_data[(training_data.item_price < 20000) & (training_data.item_cnt_day < 1000)]
training_data = training_data[training_data.item_price > 0]
training_data = training_data[training_data.item_price > 0].reset_index(drop=True)
training_data.loc[training_data.item_cnt_day < 1,'item_cnt_day'] = 0
training_data.describe()

# Data profiling for training data
training_data['item_cnt_day'] = training_data['item_cnt_day'].clip(0,20)
print("training_data's shape: {}".format(training_data.shape))
print(training_data.info())
training_data.describe()

# Data profiling for test data
print("test data's shape: {}".format(test_data.shape))
print(test_data.info())
test_data.describe()

# Data profiling for data of shops
print("Shops' shape: {}".format(shops.shape))
print(shops.info())
shops.describe()

# Data profiling for data of items
print("Items' shape: {}".format(items.shape))
print(items.info())
items.describe()

# Sample of Data sets
training_data.head(10)
training_data.sample(20)
test_data.sample(30)
shops.sample(10)
items.sample(10)

# Check whether there is any null value in the data sets
def missing_value(df):
    for col in df.columns.tolist():
        if df[col].isnull().sum() > 0:
            print("Column {} has missing {} values".format(col,df[col].isnull().sum()))
        
missing_value(training_data)
missing_value(test_data)
missing_value(shops)
missing_value(items)

#Identify & remove duplicate data
# Ref: https://www.kaggle.com/kyakovlev/1st-place-solution-part-1-hands-on-data/
training_data = training_data[training_data.duplicated() == False]
print("training_data's NEW shape: {}".format(training_data.shape))
training_data.describe()

# add a new column of profit (gain or loss) by selling the product to the training_data 
training_data_with_profit = training_data.assign(profit = training_data["item_price"]*training_data["item_cnt_day"]) #.assign(month = training_data['date_block_num']%12)
training_data_with_profit.head(10)

# Ref: https://www.kaggle.com/szhou42/predict-future-sales-top-11-solution
training_data_grouped = training_data_with_profit.groupby(['shop_id','item_id','date_block_num'])
training_data_grouped = training_data_grouped.agg({'item_cnt_day':'sum','item_price':'mean'}).reset_index()
training_data_grouped['item_cnt_day'].fillna(0)
training_data_grouped['item_cnt_day'] = training_data_grouped['item_cnt_day'].clip(0,20)
training_data_grouped.sample(20)

# Heatmap to study the correlation between features
corr = training_data_grouped.corr()
sns.heatmap(corr,cmap='YlGnBu')

# plot to study the trend of accumulated sale for each month
training_data_grouped_shop_month = training_data_grouped.groupby(['shop_id','date_block_num'])['item_cnt_day'].sum().reset_index()
ax = training_data_grouped_shop_month[training_data_grouped_shop_month['shop_id']==0].plot(x='date_block_num',y='item_cnt_day',figsize=(20,15),label="Sale number of shop " +str(0),kind="line",marker="o")
for i in [j for j in training_data_grouped_shop_month['shop_id'].unique().tolist() if j != 0]:
    training_data_grouped_shop_month[training_data_grouped_shop_month['shop_id']==i].plot(x='date_block_num', y='item_cnt_day', ax = ax, label="Sale number of shop " + str(i),kind="line",marker="o")
    ax.set_ylabel('Accumulated Sale Cout',fontsize=14)
    ax.set_xlabel('date_block_num',fontsize=14)
plt.legend(bbox_to_anchor=(0.6,-0.05),ncol=5)
plt.show()

# plot to study the trend of accumulated sale for each month
training_data_grouped_shop_month = training_data_grouped.groupby(['shop_id','date_block_num'])['item_cnt_day'].sum().reset_index()
ax = training_data_grouped_shop_month[training_data_grouped_shop_month['shop_id']==0].plot(x='date_block_num',y='item_cnt_day',figsize=(20,15),label="Sale number of shop " +str(0),kind="line",marker="o")
for i in [j for j in training_data_grouped_shop_month['shop_id'].unique().tolist() if j != 0]:
    training_data_grouped_shop_month[training_data_grouped_shop_month['shop_id']==i].plot(x='date_block_num', y='item_cnt_day', ax = ax, label="Sale number of shop " + str(i),kind="line",marker="o")
    ax.set_ylabel('Accumulated Sale Cout',fontsize=14)
    ax.set_xlabel('date_block_num',fontsize=14)
plt.legend(bbox_to_anchor=(0.6,-0.05),ncol=5)
plt.show()

# Data profiling of item_id and shop_id
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
a=training_data_grouped.groupby('item_id')['item_cnt_day'].sum().reset_index()
b=training_data_grouped.groupby('shop_id')['item_cnt_day'].sum().reset_index()
print('skew {}'.format(a.item_cnt_day.skew()))
print('kurt {}'.format(a.item_cnt_day.kurt()))
sns.displot(data=a,x='item_cnt_day',bins=5)

print('skew {}'.format(b.item_cnt_day.skew()))
print('kurt {}'.format(b.item_cnt_day.kurt()))
sns.displot(data=b,x='item_cnt_day',bins=5)

## Model Building
# build model using RidgeCV 
from sklearn.linear_model import RidgeCV
!pip install regressors
from regressors import stats
X, y = training_data_grouped.loc[:,['shop_id','item_id']], training_data_grouped['item_cnt_day']
clf = RidgeCV(cv=10).fit(X,y)
print(clf.score(X,y))
print(stats.coef_pval(clf,X,y))
stats.summary(clf,X,y)

result = clf.predict(test_data.iloc[:,1:3])
print(len(result))
print(max(result),min(result))
shop_list = test_data.iloc[:,1:3].index.values
output= np.array([shop_list,result])
np.savetxt("prediction_ridge.csv",output.T,fmt=["%d","%10.6E"],header="ID,item_cnt_month",delimiter=",")

# build model using LassoCV
from sklearn.linear_model import LassoCV
from regressors import stats
X, y = training_data_grouped.loc[:,['shop_id','item_id']], training_data_grouped['item_cnt_day']
clf_lassocv = LassoCV(cv=100, random_state=0).fit(X,y)
print(clf_lassocv.score(X,y))
print(stats.coef_pval(clf_lassocv,X,y))
stats.summary(clf_lassocv,X,y)

result = clf_lassocv.predict(test_data.iloc[:,1:3])
print(max(result),min(result))
shop_list = test_data.iloc[:,1:3].index.values
output= np.array([shop_list,result])
np.savetxt("prediction_lasso.csv",output.T,fmt=["%d","%10.6E"],header="ID,item_cnt_month",delimiter=",")

# poly
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X, y = training_data_grouped.loc[:,['shop_id','item_id']], training_data_grouped['item_cnt_day']

poly_score = []
for i in range(2,9,1):
    X_poly = PolynomialFeatures(degree=i).fit_transform(X)
    poly_fit = PolynomialFeatures(degree=i).fit(X_poly,y)
    linear_fit = LinearRegression().fit(X_poly,y)
    poly_score.append(linear_fit.score(X_poly,y))
    
plt.plot(poly_score,marker="o")
plt.xticks(range(0,7),labels=[str(l) for l in range(2,9,1)])
plt.xlabel("Degree")
plt.ylabel("R-squared")
plt.show()

# Fitting Model of polynomial
X_poly = PolynomialFeatures(degree=4).fit_transform(X)
linear_fit = LinearRegression().fit(X_poly,y)
print(linear_fit.score(X_poly,y))
print(stats.coef_pval(linear_fit,X_poly,y))
stats.summary(linear_fit,X_poly,y)

test_poly = PolynomialFeatures(degree=4).fit_transform(test_data.iloc[:,1:3])
result = linear_fit.predict(test_poly)
print(max(result),min(result))
shop_list = test_data.iloc[:,1:3].index.values
output = np.array([shop_list, result])
np.savetxt("Prediction_poly.csv",output.T,fmt=["%d","%10.6E"],header="ID,item_cnt_month",delimiter=",")

#knn
# ref: https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X, y = training_data_grouped.loc[:,['shop_id','item_id']], training_data_grouped['item_cnt_day']
knn_try = KNeighborsRegressor(n_neighbors = 5)
knn_try.fit(X,y)
result = knn_try.predict(test_data.iloc[:,1:3])
print(max(result),min(result))
knn_score_comparison = []
for i in range(1,11,1):
    knn_loop = KNeighborsRegressor(n_neighbors = i)
    knn_loop.fit(X,y)
    knn_score = knn_loop.score(X,y)
    knn_score_comparison.append(knn_score)
    
plt.plot(knn_score_comparison,marker="o")

# Model Fitting using the optimal neighbor number from the comparison above
knn = KNeighborsRegressor(n_neighbors = 8)
knn.fit(X,y)
print(knn.score(X,y))

result = knn.predict(test_data.iloc[:,1:3])
print(max(result),min(result))
shop_list = test_data.iloc[:,1:3].index.values
output= np.array([shop_list,result])
np.savetxt("prediction_knn.csv",output.T,fmt=["%d","%10.6E"],header="ID,item_cnt_month",delimiter=",")

# XGBoost
# ref: https://www.kaggle.com/szhou42/predict-future-sales-top-11-solution
from xgboost import XGBRegressor
X_train, y_train = training_data_grouped.loc[:,['shop_id','item_id']], training_data_grouped['item_cnt_day']
xgb = XGBRegressor(max_depth=10,num_round = 1000,min_child_weight=0.5,colsample_bytree=0.8,subsample=1,eta=0.3,seed=2)
xgb.fit(X_train,y_train,verbose=True,eval_metric="rmse")
print(xgb.score(X_train, y_train))

result = xgb.predict(test_data.iloc[:,1:3])
print(max(result),min(result))
shop_list = test_data.iloc[:,1:3].index.values
output= np.array([shop_list,result])
np.savetxt("prediction_xgb.csv",output.T,fmt=["%d","%10.6E"],header="ID,item_cnt_month",delimiter=",")
