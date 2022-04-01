# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:29:04 2022

@author: 85384
"""
#%%
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from xgboost import plot_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
# import packages
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from statsmodels.discrete.discrete_model import Probit
# import plotly.express as px
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
encoder = preprocessing.LabelEncoder()
plt.rc("font", size=14)
import seaborn as sns
# Quick value count calculator
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet
#%%
# import dataset
data = pd.read_csv(r"C:\\Users\85384\Desktop\Econ 2824\HW4\assignment4_HRemployee_attrition.csv")
print(data.shape)
#data info
data.info()

# Dataset columns
data.columns
# Dataset header
data.head()

data.columns.to_series().groupby(data.dtypes).groups
# Columns datatypes and missign values
data.info()
data.describe()

data.hist(figsize=(20,20))
plt.show()

data['Attrition'] = np.where(data['Attrition'] == 'Yes', 1, 0)
data['BusinessTravel'] = encoder.fit_transform(data['BusinessTravel'])
data['Department'] = encoder.fit_transform(data['Department']) 
data['EducationField'] = encoder.fit_transform(data['EducationField']) 
data['Gender'] = np.where(data['Gender'] == 'Male', 1, 0)
data['JobRole'] = encoder.fit_transform(data['JobRole']) 
data['MaritalStatus'] = encoder.fit_transform(data['MaritalStatus']) 
data['OverTime'] = np.where(data['OverTime'] == 'Yes', 1, 0)
data = data.drop(['Over18','DailyRate','EducationField','EmployeeCount','EmployeeNumber','HourlyRate','StandardHours',
                  'TrainingTimesLastYear'], axis=1)
data = data/data.max()

#%%

print(data.info())
Correlation_matrix = data.corr()
print(data.info())
# have the label in one column
target_column = ['Attrition'] 

# feature list
predictors = list(set(list(data.columns))-set(target_column))

# scale features to be in [0,1]
data[predictors] = data[predictors] / data[predictors].max()

# summary stats of scaled features
data.describe().transpose()

# divide data into test and training samples
X = data[predictors].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
#%%
# next we use MLPClassifier from sklearn.neural_network

# set up the model with two hidden layers where the first layer has 10 nodes and the second has 5
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 4)

# fit it to the training data
mlp.fit(X_train,y_train.ravel())

# apply it to the test data and compute the fitted values
y_pred = mlp.predict(X_test)


# asses model performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# plot the confusion matrix
mat = confusion_matrix(y_pred, y_test)
names = np.unique(y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()


# let's see if the regularized network does better
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 4, alpha=0.001)

# fit it to the training data
mlp.fit(X_train,y_train.ravel())

# apply it to the test data and compute the fitted values
y_pred = mlp.predict(X_test)


# asses model performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# plot the confusion matrix
mat = confusion_matrix(y_pred, y_test)
names = np.unique(y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()

#%%
# define our basic tree classifier
tree = DecisionTreeClassifier(random_state=0)

# fit it to the training data
tree.fit(X_train, y_train)
#tree.fit(xtest, ytest)

# compute accuracy in the test data
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# plot the tree
export_graphviz(tree, out_file="tree.dot", class_names=["Attrition", "No Attrition"], impurity=True, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# apply cost complexity pruning

# call the cost complexity command
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# for each alpha, estimate the tree
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)


# plot accuracy (in test and training) over alpha; first compute accuracy for each alpha
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

# second, plot it
plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


# estimate the tree with the optimal alpha and display accuracy
clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.004)
clf_.fit(X_train,y_train)

print("Accuracy on test set: {:.3f}".format(clf_.score(X_test, y_test)))

# plot the pruned tree
export_graphviz(clf_, out_file="tree.dot", class_names=["Attrition", "No Attrition"],
     impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

#regressor = LinearRegression()

#training the algorithm

#regressor.fit(X_train, y_train)
#To retrieve the intercept:
#print(regressor.intercept_)

#For retrieving the slope:
#print(regressor.coef_)
#y_pred = regressor.predict(X_test)
#test = pd.DataFrame({'Attrition': y_test.flatten(), 'Predicted': y_pred.flatten()})
#plt.scatter(X_test, y_test,  color='gray')
#plt.plot(X_test, y_pred, color='red', linewidth=2)
#plt.show()

# the simple probit classifier

#model = Probit(y_train, X_train).fit()
#print(model.summary())
#probit_predict = round(model.predict(X_test), 0)
#print("Probit's accuracy on test set: {:.3f}".format(accuracy_score(y_test, probit_predict)))

#%%
# scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# random forest with 100 and 1000 trees
regressor100 = RandomForestRegressor(n_estimators=100, # this is the number of trees in the forest
                                  random_state=0)  # this sets the seed to make this replicable

regressor1000 = RandomForestRegressor(n_estimators=1000, 
                                  random_state=0)  


# fit it to the training data
regressor100.fit(X_train, y_train)
regressor1000.fit(X_train, y_train)

# compute the prediction
y_pred100 = regressor100.predict(X_test)
y_pred1000 = regressor1000.predict(X_test)

# evaluate
print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred100)))
print('Root Mean Squared Error w 1000 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1000)))


# now we also change m
regressor100 = RandomForestRegressor(n_estimators=100,
                                     max_features = 1, # m (max number of features used in a given tree)
                                     random_state=0)

regressor100.fit(X_train, y_train)
y_pred100 = regressor100.predict(X_test)

print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred100)))

# feature importance plot
importance = regressor100.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importance)[::-1]
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# Boosting code
from sklearn.ensemble import AdaBoostClassifier

boosting = AdaBoostClassifier(n_estimators=1000, random_state=0, learning_rate=0.01)

boosting.fit(X_train, y_train)  
y_predboosting = boosting.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predboosting)))


# Gradient descent boosting
model = GradientBoostingRegressor(n_estimators=1000, random_state=0, learning_rate=0.01)

model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# XGBoost code
model = XGBClassifier(n_estimators=1000, random_state=0, learning_rate=0.01)
model.fit(X_train, y_train)

y_predbxg = model.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predbxg)))

















