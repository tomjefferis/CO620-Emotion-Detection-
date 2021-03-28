import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import numpy as np

dataset = pd.read_csv("./processed_data/completeData.csv")
dataset2 = pd.read_csv("./processed_data/completeDatabinaryarous.csv")
X = dataset.drop('labels', axis=1)
y = dataset2['labels']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

C_range = 10. ** np.arange(-12, 12)
gamma_range = 10. ** np.arange(-12, 12)

param_grid = dict(gamma=gamma_range, C=C_range,kernel=['rbf'],)


clf = SVC()


grid = GridSearchCV(estimator=clf, param_grid=param_grid,n_jobs=16)
grid.fit(X_train, y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)


