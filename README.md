# Regression-Predict-Future-Sales
## Introduction ##
The code in this repo is trying to modeling the competition "Predict Future Sales" (https://www.kaggle.com/c/competitive-data-science-predict-future-sales) [1] on Kaggle. The goal of the competition is to use historical data of sale number of on different items from different shops to predict the sale in next month (Nov 2015). This is a supervised regression problem, so various regression algorithms are used to build the model.

## Data Profiling and Data Visualization

After the data sets are imported into notebook, the training data is carefully examined using describe() to study the property of each feature. Extreme values that deviate from the mean values of the feature 'item_cnt_day' and 'item_price' are identified, as shown below.
![training_data_description](https://user-images.githubusercontent.com/30448897/147524778-1f255c43-e855-4407-a188-a1b6618e2c30.png)   
Therefore, outliers in the two columns are identified using boxplot. For example, the figure below suggests that two outlier larger than 1000 should be excluded from 'item_cnt_day' column.   
![outlier1](https://user-images.githubusercontent.com/30448897/147524974-954109b6-ea22-4eee-abae-f55e4eb5cdfd.png)   
On the other hand, 'item_price' has an outlier at around 300000.   
![outlier2](https://user-images.githubusercontent.com/30448897/147525109-949d8464-e29f-4acf-a55b-68fa62103509.png)   


## Data and Feature Engineering

## Machine Learning Model

## Results and Discussion ##
| Algorithm for Regression | Score (coefficient of determination)|
|---|---|
|Lasso with CV|6.2e-3|
|Ridge with CV|6.3e-3|
|Polynomial Regression|1.4e-2|
|KNN|3.7e-1|
|XGBoost|2.9e-1|

## Reference ##
1. Predict Future Sales https://www.kaggle.com/c/competitive-data-science-predict-future-sales 
2. 
