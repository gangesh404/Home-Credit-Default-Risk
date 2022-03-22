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


![Kaggle Score](https://user-images.githubusercontent.com/66409831/159491145-9542aa52-795e-4605-98de-4a070cbce7d6.JPG)


**References:**
https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5
https://www.kaggle.com/cloycebox/default-risk-model-week-4
https://medium.com/analytics-vidhya/home-credit-loan-default-risk-7d660ce22942
https://medium.com/analytics-vidhya/credit-default-prediction-based-on-machine-learning-models-1717601600c9
https://medium.com/@thewingedwolf.winterfell/response-coding-for-categorical-data-7bb8916c6dc1
https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/

