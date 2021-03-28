from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
# logistic regression for feature importance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("./processed_data/ecg.csv")
dataset2 = pd.read_csv("./processed_data/completeDatabinary.csv")
X = dataset#.drop('labels', axis=1)
y = dataset2['labels']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


features = X.columns.values
svm = SVC(kernel='linear')
svm.fit(X, y)

g = pd.Series(abs(svm.coef_[0]), index=features).nlargest(30)

lowest = g.index.values
print(g)