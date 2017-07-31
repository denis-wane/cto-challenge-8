'''
Created on Jul 29, 2017

@author: denis.r.wane
'''
# Check the versions of libraries

# Python version
import sys
# scipy
import scipy
# numpy
import numpy as np
# matplotlib
import matplotlib 
# pandas
import pandas as pd
# scikit-learn
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
from csv import reader 
import csv as csv
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import (metrics, cross_validation, linear_model, preprocessing)

SEED = 42  # always use a seed for randomized procedures

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

modelRF =RandomForestClassifier(n_estimators=1999, max_features='sqrt', max_depth=None, min_samples_split=9, random_state=SEED)#8803
modelXT =ExtraTreesClassifier(n_estimators=1999, max_features='sqrt', max_depth=None, min_samples_split=8,  random_state=SEED) #8903
modelGB =GradientBoostingClassifier(n_estimators=50, learning_rate=0.20, max_depth=20, min_samples_split=9, random_state=SEED)  #8749

path = ".\contest-train.csv"

base = pd.read_csv(path,header=0, encoding="utf-8")

cols_to_drop = ['UID', 'ID', 'Course URL / Code', 'Certification URL','Consolidated Course Name', 'Assigned To', 'Request Status', 'Start Date', 'End Date', 'Start Mo/Yr', 'End Mo/Yr', 'Start FY', 'End FY', 'Individual Travel Hours', 'Rqst Tot Labor Hrs', 'Airfare', 'Hotel', 'Per Diem', 'Other', 'Estimated Individual Travel', 'Misc Expenses', 'Catering', 'Facility Rental', 'Direct Other Expenses', 'Describe Other Expenses', 'Direct Expense Impact', 'Rqst NPR Alloc', 'Rqst NPR OH', 'Cancel No Response', 'Created', 'Retroactive Start Date', 'Duplicates', 'Reporting Status']
base = base.drop(cols_to_drop, axis=1)

train, test =  train_test_split(base, test_size = .2)

categorical_columns = ['Training Source', 'Home Office/Metro Area', 'Organization Number', 'Organization', 'Capability', 'Function 2', 'Career Level', 'Function', 'Function Name', 'Title','Training Type', 'Training Provider', 'Training Delivery Type', 'Training Location', 'Course Name', 'Vendor Name', 'Conference Name', 'Course or Event Name', 'Certification Type', 'Certification Name', 'Is there a course with this certification?', 'Activity', 'Support Group', 'Business Justification', 'What % of the conference is business development?', 'Travel Required']

x_train = train.drop(['Category'], axis = 1)
y_train = pd.DataFrame(data=train, columns = ['Category'])

x_test = test.drop(['Category'], axis = 1)
y_test = pd.DataFrame(data=test, columns = ['Category'])

train_objs_num = len(x_train)

x_dataset = pd.concat([x_train,x_test], axis = 0)
x_dataset_preprocessed = pd.get_dummies(data = x_dataset, columns = categorical_columns, drop_first = True)

x_train_preprocessed = x_dataset_preprocessed[:train_objs_num]
x_test_preprocessed = x_dataset_preprocessed[train_objs_num:]

y_dataset = pd.concat([y_train,y_test], axis = 0)

categories = y_dataset['Category'].unique()
category_map = dict()
i = 0
for cat in categories:
    category_map [cat] = i
    i  = i+1

y_dataset_preprocessed = y_dataset['Category'].map(category_map)

# y_dataset_preprocessed = pd.get_dummies(data = y_dataset, columns = ['Category'])

y_train_preprocessed = y_dataset_preprocessed[:train_objs_num]
y_test_preprocessed = y_dataset_preprocessed[train_objs_num:]

# # train = train.drop(categorical_columns, axis = 1)
# # test = test.drop (categorical_columns, axis = 1)
# # 
# # train.join(dummies_train)
# # test.join (dummies_test)
# 
# # cat_train_dict = train.drop( cols_to_drop, axis = 1 ).to_dict( orient = 'records' )
# # cat_test_dict = test.drop(cols_to_drop, axis = 1).to_dict(orient = 'records')
# 
# # numeric x
# numeric_cols = [ 'Training Hours', 'Certification Fee', 'Conference Fee', 'Course Fee', 'Vendor Charge' ]
# x_num_train = train[ numeric_cols ].as_matrix()
# x_num_test = test[ numeric_cols ].as_matrix()
# 
# # scale to <0,1>
# max_train = np.amax( x_num_train, 0 )
# max_test = np.amax( x_num_test, 0 )        # not really needed
# 
# x_num_train = x_num_train / max_train
# x_num_test = x_num_test / max_train        # scale test by max_train
# 
# print (train.head(3))
# 
# # y
# y_cat_train = train['Category']
# y_cat_test = test['Category']
# 
# print (y_cat_train[2])
# 
# #y_train = train.Category
# #y_test = test.Category
# 
# # categorical
# # cat_train = train.drop( numeric_cols + ['Category'], axis = 1 )
# # cat_test = test.drop( numeric_cols + ['Category'], axis = 1 )
# # 
# # cat_train.fillna( 'NA', inplace = True )
# # cat_test.fillna( 'NA', inplace = True )
# # 
# # x_cat_train = cat_train.to_dict( orient = 'records' )
# # x_cat_test = cat_test.to_dict( orient = 'records' )
# 
# # vectorize
# # vectorizer = DV( sparse = False )
# # vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
# # vec_x_cat_test = vectorizer.transform( x_cat_test )
# # 
# # vec_y_cat_train = vectorizer.fit_transform(y_cat_train)
# # vec_y_cat_test = vectorizer.fit_transform(y_cat_test)
# 
# # complete x
# x_train = train.drop(['Category'], axis = 1)
# x_test = test.drop(['Category'], axis = 1)
# 
# # complete y
# y_train = y_cat_train
# y_test = y_cat_test

# SVM looks much better in validation

print ('training SVM...')
    
# although one needs to choose these hyperparams
C = 173
gamma = 1.31e-5
shrinking = True

probability = True
verbose = True

x_train_preprocessed.to_csv('x_train_preprocessed.csv')
x_test_preprocessed.to_csv('x_test_preprocessed.csv')
y_train_preprocessed.to_csv('y_train_preprocessed.csv')
y_test_preprocessed.to_csv('y_test_preprocessed.csv')

svc = SVC( C = C, gamma = gamma, shrinking = shrinking, probability = probability, verbose = verbose )
svc.fit( x_train_preprocessed, y_train_preprocessed )
p = svc.predict_proba( x_test_preprocessed )    

print (p)
    
auc = AUC( y_test_preprocessed, p[:,1] )
print ('SVM AUC', auc)

# Load a CSV file
# def load_csv(filename):
#     file = open(filename, "rb")
#     lines = reader(file)
#     dataset = list(lines)
#     return dataset
#        
# # Convert string column to integer
# def str_column_to_int(dataset, column):
#     try:
#         class_values = [row[column] for row in dataset]
#         unique = set(class_values)
#         lookup = dict()
#         for i, value in enumerate(unique):
#             lookup[value] = i
#         for row in dataset:
#             row[column] = lookup[row[column]]
#     except:
#         return lookup
#     return lookup
# 
# dataset = load_csv(path)
# convertlist = []
# columnname = []
# 
# for i in range(len(dataset[0])):
#     columnname.append(dataset[0][i])
# 
# # convert class column to int - create int mapping array
# for i in range(len(dataset[0])):
#     lookup = str_column_to_int(dataset, i)
#     convertlist.append(lookup)
# 
# # Load dataset
# namelist = ["UID", "ID", "Training Type", "Training Provider", "Training Delivery Type", "Category"]

df = pd.read_csv(path, header=0, usecols = [5,6,10,26])

cols_to_transform = ['Organization','Capability','Function Name']
df_with_dummies = pd.get_dummies(df, columns = cols_to_transform)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print (df_with_dummies.shape)

datasetx= df_with_dummies.values 
datasety = df.values

X = datasetx[:,0:3]
Y = datasety[:,3]

print (X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print (dummy_y)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=8, activation='relu'))
    model.add(Dense(31, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=4, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

print (X.shape)
print (dummy_y.shape)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# for column in df:
#     try:
#         ind = columnname.index(column)
#         df[column].replace(convertlist[ind],inplace = True)
#     except:
#         print (column)
#         print (convertlist[ind])
#         continue
# 
# print (df.head(5))

#corr_df = df.corr(method='pearson')
#print("--------------- CORRELATIONS ---------------")
#print(corr_df.head(len(dataset)))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('Category').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist(column='Category')
#plt.show()

# scatter plot matrix
scatter_matrix(df)
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
