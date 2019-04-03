print("hello")
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
#from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

def blight_model():
    
    
    X = pd.read_csv('C:/Users/sc230m/Desktop/personal/coursera/Course4/train.csv',encoding = 'ISO-8859-1',low_memory=False)
    y = pd.read_csv('C:/Users/sc230m/Desktop/personal/coursera/Course4/test.csv',encoding = 'ISO-8859-1',low_memory=False)
  
    addresses = pd.read_csv('C:/Users/sc230m/Desktop/personal/coursera/Course4/addresses.csv',encoding = 'ISO-8859-1',low_memory=False)
    latlons = pd.read_csv('C:/Users/sc230m/Desktop/personal/coursera/Course4/latlons.csv',encoding = 'ISO-8859-1',low_memory=False)
    
    X = pd.merge(X, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    y = pd.merge(y, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    
    X.set_index('ticket_id',inplace=True)
    y.set_index('ticket_id',inplace=True)
    
    #print("Before Columns \n", X.columns)
    train_columns_del =['agency_name', 'inspector_name', 'violator_name',
       'violation_street_number', 'violation_street_name',
       'violation_zip_code', 'mailing_address_str_number',
       'mailing_address_str_name', 'city', 'state', 'zip_code',
       'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date',
       'violation_description', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'payment_amount', 'balance_due',
       'payment_date', 'payment_status', 'collection_status',
       'grafitti_status', 'compliance_detail','address','disposition'] 
    #, 'address'
    X.drop(train_columns_del,axis=1,inplace=True)
    
    test_columns_del =['agency_name', 'inspector_name', 'violator_name',
       'violation_street_number', 'violation_street_name',
       'violation_zip_code', 'mailing_address_str_number',
       'mailing_address_str_name', 'city', 'state', 'zip_code',
       'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date',
       'violation_description', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'grafitti_status','address','disposition'] 
    #'address'
    y.drop(test_columns_del,axis=1,inplace=True)
       
    #print("before na",X.size)
    #X = X.dropna(axis=0)
    #print("after na",X.size)
    
   
    valid_val=[0,1]
    
    X_clean_data = X[X.compliance.isin(valid_val)]
    
    
    labelEncoder = preprocessing.LabelEncoder()
    # X_clean_data.loc[:,'disposition']= labelEncoder.fit_transform(X_clean_data.loc[:,'disposition'])
    #y.loc[:,'disposition']= labelEncoder.fit_transform(y.loc[:,'disposition'])
    
    #X_clean_data.loc[:,'address']= labelEncoder.fit_transform(X_clean_data.loc[:,'address'])
    #y.loc[:,'address']= labelEncoder.fit_transform(y.loc[:,'address'])
    
    X_clean_data.loc[:,'violation_code']= labelEncoder.fit_transform(X_clean_data.loc[:,'violation_code'])
    y.loc[:,'violation_code']= labelEncoder.fit_transform(y.loc[:,'violation_code'])
    
    X_clean_data.loc[:,'lat'].fillna(0,inplace=True)
    X_clean_data.loc[:,'lon'].fillna(0,inplace=True)
    
    #X_clean_data.loc[:,'address'].fillna(0,inplace=True)
    
    y.loc[:,'lat'].fillna(0,inplace=True)
    y.loc[:,'lon'].fillna(0,inplace=True)
    #y.loc[:,'disposition'].fillna(0,inplace=True)
    y.loc[:,'violation_code'].fillna(0,inplace=True)
    
   
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean_data.iloc[:,X_clean_data.columns != "compliance"],X_clean_data.loc[:,"compliance"], random_state=0,test_size=0.25)
    
    #fig,subaxes = plt.subplots(1,2,figsize=(9,3))
    
    #scaler = MinMaxScaler()
    
   
    #X_train = scaler.fit_transform(X_train,y_train)
    
    #y_train = scaler.fit_transform(y_train)
    #y_score_lr = LogisticRegression(C=1.0,max_iter=100).fit(X_train, y_train)
    #y_score_lr = Ridge(alpha=20.0).fit(X_train, y_train)
    
    #print("Test1")
    #this_C = 6
    #y_score_lr = LinearSVC(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=100).fit(X_train, y_train)
    #rbf
    print("Test0")
    #X_train=X_train.iloc[0:1000]
    #y_train=y_train.iloc[0:1000]
    #,gamma='scale'
    
    
    print("gamma - scale")
    knn = KNeighborsClassifier(n_neighbors = 5)
    #y_score_lr = SVC(kernel = 'linear', probability=True,C=this_C).fit(X_train.iloc[0:1000],y_train.iloc[0:1000])
    y_score_lr = knn.fit(X_train,y_train)
    #title = 'Linear SVC, C = {:.3f}'.format(this_C)
   # plot_class_regions_for_classifier_subplot(y_score_lr, X_train, y_train, None, None, title, subaxes)
    #print("Test2")
    #y_score_lr = Ridge(alpha=20.0).fit(X_train, y_train)
    #print("Test01")
    #y.set_index('ticket_id',inplace=True)
    #yy = scaler.transform(y)
    #print("X_train_scaled",X_train)
    #y=scaler.transform(y)
    #print("after y",y)
    #print("Test")
    data_result = pd.DataFrame(y_score_lr.predict_proba(y)[:,1],y.index.values)
    #print("Test1",data_result)
    print(data_result.loc[:,0].mean())
    return  data_result # Your answer here

blight_model()
