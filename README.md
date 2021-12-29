# Regression-Predict-Future-Sales
## Introduction ##
The code in this repo is trying to modeling the competition "Predict Future Sales" (https://www.kaggle.com/c/competitive-data-science-predict-future-sales) [1] on Kaggle. The goal of the competition is to use historical data of sale number of on different items from different shops to predict the sale in next month (Nov 2015). This is a supervised regression problem, so various regression algorithms are used to build the model.

## Data Engineering, Profiling and Visualization

After the data sets are imported into notebook, the training data is carefully examined using describe() to study the property of each feature. Extreme values that deviate from the mean values of the feature 'item_cnt_day' and 'item_price' are identified, as shown below.   
![training_data_description](https://user-images.githubusercontent.com/30448897/147524778-1f255c43-e855-4407-a188-a1b6618e2c30.png)   
Therefore, outliers in the two columns are identified using boxplot. For example, the figure below suggests that two outlier larger than 1000 should be excluded from 'item_cnt_day' column.   
![outlier1](https://user-images.githubusercontent.com/30448897/147524974-954109b6-ea22-4eee-abae-f55e4eb5cdfd.png)   
On the other hand, 'item_price' has an outlier at around 300000.   
![outlier2](https://user-images.githubusercontent.com/30448897/147525109-949d8464-e29f-4acf-a55b-68fa62103509.png)   
### Data Engineering ###
After removing the outliers, there are 6 duplicate entries identified, and I discard them as the trainging set is large enough (more than 2.9M entries). The negative sale numbers (i.e. returns) are set to 0 to simplify the further reading of the result.
To study the accumulated sale numbers, the daily sale numbers for each product are summed up and grouped by shop and month, i.e. we obtain a new feature, the monthly sales number, for each product. The item_price for each aggregated sale number is replaced by the mean of prices.
### Data Visualization ###
The correlation between the features in the engineered training data is shown below, and we can fiind no significant correlation between each feature.   
![heatmap](https://user-images.githubusercontent.com/30448897/147605169-64ebda1b-7b69-48d4-ba70-2bd1eabcc08c.png)   
The trend of sale numbers across the months in data set for each shop is also shown below. Due to the large number of shops, the figure doesn't seem to provide much trend or correlation among shops, but two peaks of sales around month 11 and 23 can still be identified.     
![trend](https://user-images.githubusercontent.com/30448897/147621197-3eaacc27-f537-411a-8fc1-27b386745743.png)   
The probability distribution of 'item_cnt_day' (the monthly sale number) grouped by 'item_id' and 'shop_id', as shown below, and their skewness and kurtosis are summarized in the table. The result suggests that both distributions are really skewed, and the value is highly concentrated for 'item_cnt_day' when data is grouped by 'item_id'.   
| item_id | shop_id |
|---|---|
|![dataprofile1](https://user-images.githubusercontent.com/30448897/147623015-b4930e32-9741-433a-9991-d9035c2e4577.png)|![dataprofile2](https://user-images.githubusercontent.com/30448897/147623025-c7be18f8-f5e1-4e50-ab78-820ce56bc044.png)|
|<ul><li>skewness: 16.02</li><li>kurtosis: 576.34</li></ul>|<ul><li>skewness: 2.05</li><li>kurtosis: 5.49</li></ul>|

## Machine Learning Model
To predict the future sale, a few Machine Learning algorithms have been picked for regressional training, including two linear ones (Ridge and Lasso with CV), Polynomial regression, K-nearest Neighbors (KNN) algorithm, and Extreme Gradient Boosting (XGBoost)[2]. Linear regression models are picked as the baseline model because of the simplicity. Non-linear algorithms are also added to the inventory as linear models cannot account for enough variation to provide a better model.
For KNN, several iteration of fitting against different neighbor numbers have been run to determine the best number of neighbor (shown below), where R^2 (coefficient of determination) has been used as the metrics.   
![knn_comparison](https://user-images.githubusercontent.com/30448897/147640005-798b1bce-d0b6-4493-9a55-b338044d070f.png)   
Similarly, for Polynomial regression, multiple iterations have been run to determine the number of degree which can account for the largest variation, as shown below.    
![poly_comparison](https://user-images.githubusercontent.com/30448897/147640078-b70a91c5-e522-4c12-beda-3fdd28d630ce.png)   

## Results and Discussion ##
The features 'shop_id' and 'item_id' in training data as the independent variables while using 'item_cnt_day' as the dependent variable, without further feature engineering. The test set also contains the features 'shop_id' and 'item_id'.   
The results obtained from various algorithm are summarized in the table below. One can find the trend that in general, high the R^2 is, the lower the score (Root Mean Square Error, RMSE) is, except the KNN model. This trend is expected as the original training data is really scattered, so the model that can account for larger portion of variation should be a better model. (The current models are still distant away from overfitting) The summary suggests that XGBoost algorithm can provide the best model in this problem.      
| Algorithm for Regression | R^2 (coefficient of determination)| Submission Score (RMSE) |
|---|---|---|
|Lasso with CV|6.2e-3|2.119|
|Ridge with CV|6.3e-3|2.118|
|Polynomial Regression (Degree=4)|1.4e-2|2.0851|
|KNN (N=8)|3.7e-1|2.150|
|XGBoost|2.9e-1|2.023|

The assumption behind the reasion why KNN devaites from the trend is that the two input features, 'shop_id' and 'item_id' are not meaningful feature, i.e. one can replace the current shop_id or item_id with any numerical value, so when KNN tries to minimize distance between any given output data point, a new pair of shop_id and item_id could generate a extreme value with large contribution to RMSE.

## Reference ##
1. Predict Future Sales https://www.kaggle.com/c/competitive-data-science-predict-future-sales 
2. XGBoost https://www.nvidia.com/en-us/glossary/data-science/xgboost/
