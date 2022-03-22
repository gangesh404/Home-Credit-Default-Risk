# Home-Credit-Default-Risk
## -by Gangesh Vaniyamkandi
Predicting the capability of each applicant of repaying a loan

### Introduction:
Home Credit B.V. is an international non-bank financial institution who provides loan for a varied personal requirement. They require a well optimized machine learning approach to determine whether an applicant will pay back the loan or not.

### Business Problem:
As many loan applicants face rejections due to very less or no credit history and end up with untrustworthy lenders. In order to solve this problem, we need to determine if an applicant would be able to repay the loan back using the available data. So that the client has a better borrowing experience and the Home Credit can avoid any defaulters and make the most profit by lending the money to the ideal applicant.

### ML formulation of business problem:
For each applicant we need to predict if they can repay the loan. Here the targets are 0 and 1, where 0 means the applicant will repay the loan and 1 means the applicant will not repay the loan. This is a binary classification problem.

### Business constraints: 
In this problem finding the ideal applicant without misclassification is very important because misclassification will cause huge loss for the Home Credit and also the applicant will face a negative loan experience. We do not have any low latency problem as loan approval process usually takes some days.

### The Data Set consists of –
**a) application_{train|test}.csv –** This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET). Static data for all applications. One row represents one loan in our data sample.

**b) bureau.csv -** All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample). For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

**c) bureau_balance.csv -** Monthly balances of previous credits in Credit Bureau. This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

**d) POS_CASH_balance.csv -** Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

**e) credit_card_balance.csv** - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

**f) previous_application.csv -** All previous applications for Home Credit loans of clients who have loans in our sample. There is one row for each previous application related to loans in our data sample.

**g) Installments_payments.csv -** Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample. There is a) one row for every payment that was made plus b) one row each for missed payment. One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

## First Cut Approach:

1. Data Preprocessing – We need to merge all the csv files into application_train data by aggregation techniques. Then we perform EDA and remove the outliers. For this we first replace the outliers with nan value and later impute them with zero in case of numerical features and mode in case of categorical features. Features having outliers more than 75 percent can be dropped. Next, we need to plot the correlation plot for all the data to find the features which are mutually correlated and keep only one of them and drop the other one.

2. Perform Feature Engineering

3. Next, we need to normalize all the numerical data and response encoding all the categorical data and save all the normalizer and response encoding fit.

4. Fit on LGBMClassifier and get the best featuers list

5. Perform GridSearch CV on LightGBM Classifier and find the best hyperparameters.

6. Keep only the features with good feature importance value and drop the others.

7. Fit LigthGBM with best hyperparameters on the updated dataset and save the model. Plot the confusion matrix and AUC curve for the test and train score.

8. Use the saved StandardScalar, Response Encoding, One_hot_encoding and best model and pre-process and predict the test data.

## Below are the Jupyter notebooks with the scripts and  and explanations:

 - [Exploratory Data Analysis](https://github.com/gangesh404/Home-Credit-Default-Risk/blob/main/Home_Credit_EDA.ipynb)
 - [Preprocessing and Model training](https://github.com/gangesh404/Home-Credit-Default-Risk/blob/main/Home_Credit_Feature_Engg_and_Training.ipynb)
 - [Predict Targets on Test Dataset](https://github.com/gangesh404/Home-Credit-Default-Risk/blob/main/Home_Credit_Test.ipynb)

## Below are the Top 30 Features:

![best_features](https://user-images.githubusercontent.com/66409831/159518785-c2671336-ed86-4715-b7d8-1b08ebfcf275.png)

## Results on Submission on Kaggle:

![Kaggle Score] https://user-images.githubusercontent.com/66409831/159491145-9542aa52-795e-4605-98de-4a070cbce7d6.JPG


**References:**
https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5

https://www.kaggle.com/cloycebox/default-risk-model-week-4

https://medium.com/analytics-vidhya/home-credit-loan-default-risk-7d660ce22942

https://medium.com/analytics-vidhya/credit-default-prediction-based-on-machine-learning-models-1717601600c9

https://medium.com/@thewingedwolf.winterfell/response-coding-for-categorical-data-7bb8916c6dc1

https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/

