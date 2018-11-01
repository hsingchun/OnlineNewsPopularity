#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:15:30 2018

@author: celiachen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import statistics as st
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import time

""" 
Step0: Problem Statement
This is a dataset about online news popularity from UCI.
[Topic 1]
Business Goal: To estimate how popular the future news will be, as a reference for advertisting 
Analytic Goal: Predict number of shares of the news proposed before it goes online
               (Regression, Adaboost, RegressinTree )

""" 


"""
Step1: Load the online news dataset
"""
data = pd.read_csv("OnlineNewsPopularity.csv")
data.head()

"""
Step2: EDA
"""
    # Get statistics of all original attributes
data.isnull().sum().sum()
data[' shares'].describe()
data.describe()
#use 50%: 1400 as cutoff of popular or not


    # Visualize the feature of different day of week
columns_day = data.columns.values[list(data.columns).index(' weekday_is_monday'):list(data.columns).index(' weekday_is_sunday')+1]
unpop=data[data[' shares']<1400]
pop=data[data[' shares']>=1400]
unpop_day = unpop[columns_day].sum().values
pop_day = pop[columns_day].sum().values
    #figure: days
pl.figure(figsize = (13,5))
pl.title("Count of popular/unpopular news over different day of week", fontsize = 16)
pl.bar(np.arange(len(columns_day)), pop_day, width = 0.3, align="center", color = 'gold',label = "popular")
pl.bar(np.arange(len(columns_day)) - 0.3, unpop_day, width = 0.3, align = "center", color = 'g',label = "unpopular")
pl.xticks(np.arange(len(columns_day)), columns_day)
pl.ylabel("Count", fontsize = 9)
pl.xlabel("Days of week", fontsize = 12)
pl.legend(loc = 'upper right')
pl.tight_layout()
pl.savefig("days.pdf")

    # Visualize the feature of different channels
columns_channel = data.columns.values[list(data.columns).index( ' data_channel_is_lifestyle'):list(data.columns).index( ' data_channel_is_world')+1]
unpop_channel = unpop[columns_channel].sum().values
pop_channel = pop[columns_channel].sum().values
    #figure: channel 
pl.figure(figsize = (13,5))
pl.title("Count of popular/unpopular news over channels", fontsize = 16)
pl.bar(np.arange(len(columns_channel)), pop_channel, width = 0.3, align="center", color = 'gold',label = "popular")
pl.bar(np.arange(len(columns_channel)) - 0.3, unpop_channel, width = 0.3, align = "center", color = 'g',label = "unpopular")
pl.xticks(np.arange(len(columns_channel)), columns_channel)
pl.ylabel("Count", fontsize = 9)
pl.xlabel("Channel", fontsize = 12)
pl.legend(loc = 'upper right')
pl.tight_layout()
pl.savefig("channel.pdf")


"""
Step3: Data Preprocessing
"""
# tackle: extreme values of target outcome 
    # Visualize the distribution of outcome target
#share_ori= sns.distplot(data[' shares'],hist=True, kde=False,hist_kws={'edgecolor':'black'})
data[' shares'].plot(kind='hist', bins=100)
pl.title("Original Shares_distribution", fontsize = 16)
pl.ylabel("Count", fontsize = 9)
pl.xlabel("Number of shares", fontsize = 9)
pl.savefig("Original_Shares_distribution.pdf")

#pl.boxplot(data[' shares'])

nonextre=data[data[' shares']< st.mean(data[' shares'])+ 2*st.stdev(data[' shares'])]
#sns.distplot(nonextre[' shares'],hist=True, kde=False,hist_kws={'edgecolor':'black'})
nonextre[' shares'].plot(kind='hist', bins=100)
pl.ylabel("Count", fontsize = 9)
pl.xlabel("Number of shares", fontsize = 9)
pl.title("95% of Shares_distribution", fontsize = 16)
pl.savefig("95percent_Shares_distribution.pdf")
#pl.boxplot(nonextre[' shares'])

''' df : del 'url' and add 'log_shares from data' '''
df= data.drop('url',1)
df['log_shares']=np.log(data[' shares'])
original_df_names=list(df.columns.values)
len(original_df_names)

#sns.distplot(data['log_shares'],hist=True, kde=False,hist_kws={'edgecolor':'black'})
pl.title("Log of Shares_distribution", fontsize = 16)
df['log_shares'].plot(kind='hist', bins=100)
pl.ylabel("Count", fontsize = 9)
pl.xlabel("Number of shares", fontsize = 9)
pl.savefig("Log of Shares_distribution.pdf")
#pl.boxplot(data['log_shares'])



    #use log: data['log_shares']
from sklearn.preprocessing import MinMaxScaler
# normalization of numerical attribues
scaler = MinMaxScaler()
numerical = [' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
        ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs',
        ' num_self_hrefs', ' num_imgs', ' num_videos',
        ' average_token_length', ' num_keywords',' kw_min_min',
        ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max',
        ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg',
        ' self_reference_min_shares', ' self_reference_max_shares',
        ' self_reference_avg_sharess', 
        ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04',
        ' global_subjectivity', ' global_sentiment_polarity',
        ' global_rate_positive_words', ' global_rate_negative_words',
        ' rate_positive_words', ' rate_negative_words',
        ' avg_positive_polarity', ' min_positive_polarity',
        ' max_positive_polarity', ' avg_negative_polarity',
        ' min_negative_polarity', ' max_negative_polarity',
        ' title_subjectivity', ' title_sentiment_polarity',
        ' abs_title_subjectivity', ' abs_title_sentiment_polarity']
data[numerical] = scaler.fit_transform(data[numerical])
    #data[numerical].describe()

# multicollinearity test

corr_matrix = df.corr().abs()
high_corr_var=list(np.where(corr_matrix>0.8))
#high_corr_ex=[(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
i=0
high_corr=[]
for x in range(len(high_corr_var[0])):
    if high_corr_var[0][x] != high_corr_var[1][x] and high_corr_var[0][x] < high_corr_var[1][x]:
        high_corr.append([list(df.columns)[high_corr_var[0][x]],list(df.columns)[high_corr_var[1][x]]])
        i=1+1
        
''' del multicollinear'''
#df=df.drop( [' weekday_is_saturday', ' weekday_is_sunday', ' is_weekend', ' shares',' self_reference_min_shares',' self_reference_max_shares'],1)

"""
[Topic 1] Predict number of shares of the news proposed before it goes online
               (Regression, Adboost, NN, RegressinTree )

Step3: Data Preprocessing
"""
# co-linear test
df_pre=df.drop( [' weekday_is_saturday', ' weekday_is_sunday', ' is_weekend', 
                 ' shares',' self_reference_min_shares',' self_reference_max_shares',
                 ' n_non_stop_words',' n_non_stop_unique_tokens',' kw_min_avg',
                 ' kw_max_avg', ' kw_avg_avg',' kw_min_min', ' kw_max_min', ' kw_avg_min',],1)
list(df_pre.columns)
features_pre = df_pre.drop('log_shares',1)

# Encode the label by threshold 1400 (50%)

outcome_label =pd.Series(label_encoder.fit_transform(data[' shares']>=1400))

"""
[Topic 1] Predict number of shares of the news proposed before it goes online
               (Regression, KNN, NN, RegressinTree )

Step4: Modeling
"""

# feature selection
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

time_start = time.clock()# time start

estimator = AdaBoostClassifier(random_state=0)
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(features_pre, outcome_label)
selector.ranking_

estimator_LR = LogisticRegression(random_state=0)
selector_LR = RFECV(estimator_LR, step=1, cv=5)
selector_LR = selector_LR.fit(features_pre, outcome_label)
selector_LR.ranking_

estimator_RF = RandomForestClassifier(random_state=0)
selector_RF = RFECV(estimator_RF, step=1, cv=5)
selector_RF = selector_RF.fit(features_pre, outcome_label)
selector_RF.ranking_

# Plot the cv score vs number of features
#---------------
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
pl.savefig('RFE_ADA.pdf')
pl.show()

features_pre.columns.values[selector.ranking_==1].shape[0]
features_pre.columns.values[selector.ranking_==1]
features_ADA = features_pre[features_pre.columns.values[selector.ranking_==1]]

#---------------
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector_LR.grid_scores_) + 1), selector_LR.grid_scores_)
pl.savefig('RFE_LR.pdf')
pl.show()

features_pre.columns.values[selector_LR.ranking_==1].shape[0]
features_pre.columns.values[selector_LR.ranking_==1]
features_LR = features_pre[features_pre.columns.values[selector_LR.ranking_==1]]

#---------------
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector_RF.grid_scores_) + 1), selector_RF.grid_scores_)
pl.savefig('RFE_RF.pdf')
pl.show()

features_pre.columns.values[selector_RF.ranking_!=1].shape[0]
features_pre.columns.values[selector_RF.ranking_!=1]
features_RF = features_pre[features_pre.columns.values[selector_RF.ranking_==1]]

#---------------
time_elapsed = (time.clock() - time_start)# time end

# data partition
from sklearn.metrics import accuracy_score, fbeta_score, roc_curve, auc, roc_auc_score
from sklearn.cross_validation import train_test_split

X_train_ADA, X_test_ADA, y_train_ADA, y_test_ADA = train_test_split(features_ADA, outcome_label, test_size = 0.2, random_state = 123)
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(features_LR, outcome_label, test_size = 0.2, random_state = 123)
X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(features_RF, outcome_label, test_size = 0.2, random_state = 123)

# modeling & evaluation
def train_predict(learner, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    start = time.clock() # Get start time
    learner.fit(X_train, y_train)
    end = time.clock() # Get end time

    results['train_time'] = end-start
        
    # Get predictions on the training samples
    start = time.clock() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time.clock() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute accuracy on the training samples
    results['acc_train'] = accuracy_score(y_train,predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
        
    # Compute F-score on the training samples
    results['f_train'] = fbeta_score(y_train,predictions_train,beta=1)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=1)
    
    # Compute AUC on the training samples
    results['auc_train'] = roc_auc_score(y_train,predictions_train)
        
    # Compute AUC on the test set
    results['auc_test'] = roc_auc_score(y_test,predictions_test)
       
    # Success
    print ("{} with accuracy {}, F1 {} and AUC {}.".format(learner.__class__.__name__,results['acc_test'],results['f_test'], results['auc_test']))
    
    # Return the results
    return results

train_predict(AdaBoostClassifier(), X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)
train_predict(LogisticRegression(), X_train_LR, y_train_LR, X_test_LR, y_test_LR)
train_predict(RandomForestClassifier(), X_train_RF, y_train_RF, X_test_RF, y_test_RF)





