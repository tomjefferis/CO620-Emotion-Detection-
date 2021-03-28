import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']



dataset = pd.read_csv("./processed_data/ecg.csv")
dataset2 = pd.read_csv("./processed_data/completeData.csv")

y = dataset2['labels']
X = dataset#.drop('labels', axis=1)

#for dataset finding lowest weighted features
features = X.columns.values
svm = SVC(kernel='linear')
svm.fit(X, y)

g = pd.Series(abs(svm.coef_[0]), index=features).nsmallest(30)
lowest = g.index.values
scores = []

#valence model
#model = SVC(C=10.0, gamma=4)
#arousal model
#model = SVC(gamma=1.5)
#arousal model with eyes
model = SVC()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)
cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
clf = model.fit(X_train, y_train)
y_validation = clf.predict(X_val)
x = classification_report(y_val, y_validation, output_dict=True)
scores.append(("baseline", x["accuracy"]))

for item in lowest:
    tempdrop = item
    X.drop(tempdrop,inplace=True,axis=1)
    models = [[tempdrop, X]]
    for label, dataset in models:
        #X = dataset.drop('labels', axis=1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_validation = clf.predict(X_val)
        x = classification_report(y_val, y_validation,output_dict=True)
        scores.append((label, x["accuracy"]))
        #scores[label] = x["accuracy"]


scores2 = pd.DataFrame(scores)
gethigh = scores2.nlargest(1,1)[0].index.values.astype(int)[0]
todrop  = lowest[:gethigh]


dataset = pd.read_csv("./processed_data/completeData.csv")
dataset2 = pd.read_csv("./processed_data/completeData.csv")

y = dataset2['labels']
X = dataset.drop('labels', axis=1)
X.drop(todrop, axis=1,inplace=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)
cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
clf = model.fit(X_train, y_train)
y_validation = clf.predict(X_val)
print(classification_report(y_val, y_validation))

