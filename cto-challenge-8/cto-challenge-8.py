'''
Created on Jul 29, 2017

@author: denis.r.wane
'''
# Check the versions of libraries

# Python version
from csv import reader 
import scipy
import sys

import matplotlib 
import pandas
from pandas.plotting import scatter_matrix
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn import model_selection
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import csv as csv
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train_path = ".\contest-train.csv"
out_path = ".\contest-test.csv"

train_base = pd.read_csv(train_path,header=0, encoding="utf-8")
out_base = pd.read_csv(out_path,header=0,encoding="utf-8")

cols_to_drop = ['UID', 'ID', 'Course Name', 'Course URL / Code', 'Certification URL','Consolidated Course Name', 'Assigned To', 'Request Status', 'Start Date', 'End Date', 'Start Mo/Yr', 'End Mo/Yr', 'Start FY', 'End FY', 'Individual Travel Hours', 'Rqst Tot Labor Hrs', 'Airfare', 'Hotel', 'Per Diem', 'Other', 'Estimated Individual Travel', 'Misc Expenses', 'Catering', 'Facility Rental', 'Direct Other Expenses', 'Describe Other Expenses', 'Direct Expense Impact', 'Rqst NPR Alloc', 'Rqst NPR OH', 'Cancel No Response', 'Created', 'Retroactive Start Date', 'Duplicates', 'Reporting Status']
categorical_columns = ['Training Source', 'Home Office/Metro Area', 'Organization Number', 'Organization', 'Capability', 'Function 2', 'Career Level', 'Function', 'Function Name', 'Title','Training Type', 'Training Provider', 'Training Delivery Type', 'Training Location', 'Vendor Name', 'Conference Name', 'Course or Event Name', 'Certification Type', 'Certification Name', 'Is there a course with this certification?', 'Activity', 'Support Group', 'Business Justification', 'What % of the conference is business development?', 'Travel Required']

train_base = train_base.drop(cols_to_drop, axis=1)
out_base = out_base.drop(cols_to_drop, axis=1)

full_set = pd.concat([train_base,out_base], axis=0)
x_full_preprocessed = pd.get_dummies(data = full_set, columns = categorical_columns, drop_first = True)

#remove rows where Category is NaN
x_train_fin = x_full_preprocessed.dropna(subset=['Category'])
#output portion should be records where Category is NaN
x_out_fin = x_full_preprocessed[x_full_preprocessed.Category.isnull()]
x_out_fin = x_out_fin.drop(['Category'], axis = 1)

#create a map for Category values
categories = x_train_fin['Category'].unique()
category_map = dict()
i = 0
for cat in categories:
    category_map [cat] = i
    i  = i+1

#split the training portion
train, test =  train_test_split(x_train_fin, test_size = .1)

#separate Category
x_train = train.drop(['Category'], axis = 1)
y_train = pd.DataFrame(data=train, columns = ['Category'])

x_test = test.drop(['Category'], axis = 1)
y_test = pd.DataFrame(data=test, columns = ['Category'])

#apply the map to Category
y_train_preprocessed = pd.DataFrame(data = y_train['Category'].map(category_map))
y_test_preprocessed = pd.DataFrame(data = y_test['Category'].map(category_map))

# Test options and evaluation metric
seed = 56
C = 8.5
scoring = 'accuracy'

# Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(model, x_train_preprocessed, y_train_preprocessed.values.ravel(), cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)


# Check for best C value to use
# for i in np.arange(0.1, 10.0, 0.2):
#     LR = LogisticRegression(C=i)
#     LR.fit(x_train_preprocessed, y_train_preprocessed.values.ravel())
#     predictions = LR.predict(x_test_preprocessed)
#     print('C= ', i, ' - ', accuracy_score(y_test_preprocessed.values.ravel(), predictions))

LR = LogisticRegression(C=C)
LR.fit(x_train, y_train_preprocessed.values.ravel())
predictions = LR.predict(x_test)    
print (accuracy_score(y_test_preprocessed.values.ravel(), predictions))

predictions_test = LR.predict(x_out_fin)
print(predictions_test)
print (predictions_test.shape)

i = 0
for row in predictions_test:
    cat_val = list(category_map.keys())[list(category_map.values()).index(row)]
    out_base.ix[i,'Category'] = cat_val
    i = i+1
    
out_base.to_csv('results.csv')    
