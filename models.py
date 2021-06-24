import time
from sklearn import metrics
from solve import _data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# create models
X_train, y_train, X_test, y_test = _data()
# print(X_test)
# logistic regression
# tang so lan lap max_iter de thuat toan tranh bi hoi tu
logis = LogisticRegression(max_iter=1000)
logis.fit(X_train, y_train)
start = time.time()
y_pred1 = logis.predict(X_test)
end = time.time()
time1 = round(end-start, 10)
score1 = metrics.accuracy_score(y_test, y_pred1) * 100
# decision tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
start = time.time()
y_pred2 = regressor.predict(X_test)
end = time.time()
time2 = round(end-start, 20)
score2 = metrics.accuracy_score(y_test, y_pred2) * 100
# Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
start = time.time()
y_pred4 = gnb.predict(X_test)
end = time.time()
time4 = round(end-start, 20)
score4 = metrics.accuracy_score(y_test, y_pred4) * 100
# SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
start = time.time()
y_pred5 = clf.predict(X_test)
end = time.time()
time5 = round(end-start, 20)
score5 = metrics.accuracy_score(y_test, y_pred5) * 100
