import matplotlib
from matplotlib import gridspec
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
from collections import Counter
import matplotlib
import pysiology as pyd
import seaborn as sns
import pandas as pd
from matplotlib import pyplot, gridspec
import neurokit2 as nk2
import matplotlib.pyplot as plt
import statistics
import heartpy as hp
import numpy as np
import scipy as sci
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import performance_metrics

def run_exps(final) -> pd.DataFrame:

    bootstraps = []
    for model in list(set(final.model.values)):
        model_df = final.loc[final.model == model]
        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')
    time_metrics = ['fit_time', 'score_time']  # fit time metrics
    ## PERFORMANCE METRICS
    results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)]  # get df without fit data
    results_long_nofit = results_long_nofit.sort_values(by='values')
    ## TIME METRICS
    results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)]  # df with fit data
    results_long_fit = results_long_fit.sort_values(by='values')

    plt.figure(figsize=(60, 24))
    plt.tight_layout()
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Classification Metric')
    plt.savefig('./results/benchmark_models_performance.png', dpi=300)

    plt.figure(figsize=(60, 24))
    plt.tight_layout()
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Fit and Score Time')
    plt.savefig('./results/benchmark_models_time.png',dpi=300)

    metrics = list(set(results_long_nofit.metrics.values))
    bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])

    time_metrics = list(set(results_long_fit.metrics.values))
    bootstrap_df.groupby(['model'])[time_metrics].agg([np.std, np.mean])
    return final


data = pd.read_csv('./processed_data/completeDataNS.csv')
X = data.drop('labels', axis=1)
y = data['labels']
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



tsne_results  = TSNE(n_components=2).fit_transform(X_train)
df_subset = pd.DataFrame()

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['tsne-2d-lab'] = y_train

plt.rcParams["figure.figsize"] = [16,9]
fig, axs = plt.subplots(11,figsize=(30,40))


colors = ['red','green','blue','yellow']
axs[10].scatter(df_subset['tsne-2d-one'],df_subset['tsne-2d-two'],c=y_train,cmap=matplotlib.colors.ListedColormap(colors))
axs[10].set_title("TSNE Data")



# Modeling step Test differents algorithms
classifiers =   [("AdaBoost",AdaBoostClassifier()),
                ("RFC",RandomForestClassifier()),
                ("KNN",KNeighborsClassifier(3)),
                #("Linear SVC", SVC(kernel="linear", C=0.025)),
                ("SVC",SVC()),
                ("GPC",GaussianProcessClassifier(1.0 * RBF(1.0))),
                ("DTC",DecisionTreeClassifier(max_depth=5)),
                ("MLP",MLPClassifier(alpha=1, max_iter=1000)),
                ("NB",GaussianNB()),
                ("QDA",QuadraticDiscriminantAnalysis())]

results = []
names = []
dfs = []
report = pd.DataFrame()
for index, data in enumerate(classifiers) :
    name = data[0]
    model = data[1]
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_mod = clf.predict(X_train)
    #print(f"{names[model]} classification report")
    cr = classification_report(y_test, y_pred,output_dict=True)
    cr["name"] = name
    report = pd.concat([report,pd.DataFrame(cr)])
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
    #results.append([name,cr])
    # create meshgrid
    resolution = 400 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(tsne_results[:,0]), np.max(tsne_results[:,0])
    X2d_ymin, X2d_ymax = np.min(tsne_results[:,1]), np.max(tsne_results[:,1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(tsne_results, y_mod)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    #plot
    axs[index].contourf(xx, yy, voronoiBackground,cmap=matplotlib.colors.ListedColormap(colors),alpha = 0.8)
    axs[index].scatter(tsne_results[:,0], tsne_results[:,1], c=y_train,cmap=matplotlib.colors.ListedColormap(colors),alpha = 0.4)
    axs[index].set_title(name)



#print(x)

fig = plt.gcf()
gs = gridspec.GridSpec(4,3)
for i in range(3):
    for j in range(4):
        k = i+j*3
        if k < len(axs):
            axs[k].set_position(gs[k].get_position(fig))


fig.savefig('./results/visual_classifier_decisions.png',dpi=400)
fig.show()
x = run_exps(final)
report.to_csv("./results/report.csv")