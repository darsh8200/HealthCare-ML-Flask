import flask
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# In this datasets no need of preprocessing as it's a classification and all are one Hot Encoder type of dataset.
training = pd.read_csv("Training.csv")
cols = training.columns
cols=cols[:-1]
X = training[cols]
y = training["prognosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

testing = pd.read_csv("Testing.csv")
testx = testing[cols]
testy = testing['prognosis']  
testy = le.transform(testy)

# Below are some classifiers which are used for different models.
dtclf = DecisionTreeClassifier()
kclf = KNeighborsClassifier()
svmclf = SVC()

# These are classifiers(clf1, clf2).
# One is DecisionTree Classifier and other one is Support Vector Machines Classifier.
'''
clf1 = dtclf.fit(X_train, y_train)
dt_score = cross_val_score(clf1, X_test, y_test, cv=3, scoring="accuracy")
print(dt_score.mean())

clf2 = svmclf.fit(X_train, y_train)
svm_score = cross_val_score(kclf, X_test, y_test, cv=3, scoring="accuracy")
print(svm_score.mean())

# Feature Importances and All.
importances = clf1.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols
#print(importances)
'''
clf = dtclf.fit(X, y)
pickle.dump(clf,open('model_F.pkl','wb'))
model = pickle.load(open('model_F.pkl','rb'))