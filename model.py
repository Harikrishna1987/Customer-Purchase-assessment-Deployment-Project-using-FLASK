"""
Author: HARIKRISHNA.MUDIDUDDI
PLACE: HYDERABAD
DESIGNATION: Jr Data Scientist
PROJECT: CUSTOMER ASSISMENT BASED ON THE PURCHASE
"""

import pandas as pd

import pickle

sv = pd.read_csv('data.csv')

sv.isnull().sum()

sv['Gender'] = sv['Gender'].replace({'Female': 0, 'Male': 1})

sv.drop(["User ID"], axis=1, inplace=True)

X = sv.iloc[:, :3].values

Y = sv.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state = 42)

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
from sklearn.svm import SVC

svclassifier = SVC()

svclassifier = svclassifier.fit(X_train, Y_train)


xpred = svclassifier.predict(X_train)
ypred = svclassifier.predict(X_test)


""" Bagging comes under Ensemble Method """
from sklearn.ensemble import BaggingClassifier

bag_mod = BaggingClassifier(n_estimators=200)

bagg = bag_mod.fit(X_train, Y_train)


x_pred = bagg.predict(X_train)

y_pred = bagg.predict(X_test)


pickle.dump(bag_mod, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))
