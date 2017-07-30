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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path = ".\contest-train.csv"

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rb")
    lines = reader(file)
    dataset = list(lines)
    return dataset
       
# Convert string column to integer
def str_column_to_int(dataset, column):
    try:
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
    except:
        return lookup
    return lookup

dataset = load_csv(path)
convertlist = []
columnname = []

for i in range(len(dataset[0])):
    columnname.append(dataset[0][i])

# convert class column to int - create int mapping array
for i in range(len(dataset[0])):
    lookup = str_column_to_int(dataset, i)
    convertlist.append(lookup)

# Load dataset
namelist = ["UID", "ID", "Training Type", "Training Provider", "Training Delivery Type", "Category"]

df = pd.read_csv(path, usecols=[5, 6, 10, 25, 26, 27, 28])

for column in df:
    try:
        ind = columnname.index(column)
        df[column].replace(convertlist[ind],inplace = True)
    except:
        print (column)
        print (convertlist[ind])
        continue

print (df.head(5))

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
