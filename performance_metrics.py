import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib import pyplot, gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# import performance_metrics


def run_exps(final, title) -> pd.DataFrame:
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

    plt.figure(figsize=(30, 12))
    plt.tight_layout()
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Classification Metric')
    plt.tight_layout()
    plt.savefig(f'./results/{title}/benchmark_models_performance.png', dpi=200)

    plt.figure(figsize=(30, 12))
    plt.tight_layout()
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Fit and Score Time')
    plt.tight_layout()
    plt.savefig(f'./results/{title}/benchmark_models_time.png', dpi=200)

    metrics = list(set(results_long_nofit.metrics.values))
    bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])

    time_metrics = list(set(results_long_fit.metrics.values))
    bootstrap_df.groupby(['model'])[time_metrics].agg([np.std, np.mean])
    return final


def scriptRunner(data, title, classifiers) -> pd.DataFrame:
    X = data.drop('labels', axis=1)
    y = data['labels']
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    tsne_results = TSNE(n_components=2).fit_transform(X_train)
    df_subset = pd.DataFrame()

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['tsne-2d-lab'] = y_train

    plt.rcParams["figure.figsize"] = [16, 9]
    fig, axs = plt.subplots(11, figsize=(15, 20))

    colors = ['red', 'green', 'blue', 'yellow']
    axs[10].scatter(df_subset['tsne-2d-one'], df_subset['tsne-2d-two'], c=y_train,
                    cmap=matplotlib.colors.ListedColormap(colors))
    axs[10].set_title("TSNE Data")

    results = []
    names = []
    dfs = []
    crm = {}
    report = pd.DataFrame()
    for index, data in enumerate(classifiers):
        name = data[0]
        model = data[1]
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_mod = clf.predict(X_train)
        y_validation = clf.predict(X_val)
        cr = classification_report(y_val, y_validation, output_dict=True)
        crm[name] = confusion_matrix(y_val, y_validation, normalize="true")
        cr["name"] = name
        report = pd.concat([report, pd.DataFrame(cr)])
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
        # results.append([name,cr])
        # create meshgrid
        resolution = 200  # 100x100 background pixels
        X2d_xmin, X2d_xmax = np.min(tsne_results[:, 0]), np.max(tsne_results[:, 0])
        X2d_ymin, X2d_ymax = np.min(tsne_results[:, 1]), np.max(tsne_results[:, 1])
        xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

        # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
        background_model = KNeighborsClassifier(n_neighbors=1).fit(tsne_results, y_mod)
        voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
        voronoiBackground = voronoiBackground.reshape((resolution, resolution))

        # plot
        axs[index].contourf(xx, yy, voronoiBackground, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.8)
        axs[index].scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_train,
                           cmap=matplotlib.colors.ListedColormap(colors), alpha=0.4)
        axs[index].set_title(name)

    # print(x)

    fig = plt.gcf()
    gs = gridspec.GridSpec(4, 3)
    for i in range(3):
        for j in range(4):
            k = i + j * 3
            if k < len(axs):
                axs[k].set_position(gs[k].get_position(fig))

    fig.savefig(f'./results/{title}/visual_classifier_decisions.png', dpi=200)
    fig.show()

    run_exps(final, title)
    report.to_csv(f"./results/{title}/report.csv")
    return crm


def confusionMatrixPlot(confusionMatrix, title, size):
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    for key, ax in zip(confusionMatrix, axes.flatten()):
        df_cm = pd.DataFrame(confusionMatrix[key], range(size), range(size))
        # plt.figure(figsize=(10,7))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ax=ax, cmap="crest")
        ax.set_title(key)  # font size

    plt.savefig(f'./results/{title}/confusionmatrix.png', dpi=200)
    plt.show()


# Modeling step Test differents algorithms
def main(results, classifiers = None):
    if classifiers is None:
        classifiers = [("AdaBoost", AdaBoostClassifier()),
                       ("RFC", RandomForestClassifier()),
                       ("KNN", KNeighborsClassifier()),
                       # ("LSVC", SVC(kernel="linear", C=0.025)),
                       ("SVC", SVC()),
                       ("GPC", GaussianProcessClassifier()),
                       ("DTC", DecisionTreeClassifier()),
                       ("MLP", MLPClassifier()),
                       ("NB", GaussianNB()),
                       ("QDA", QuadraticDiscriminantAnalysis())]
    for key in results:
        data = pd.read_csv(results[key])
        data.dropna(axis=0, inplace=True)

        labels = data["labels"].nunique()

        cm = scriptRunner(data, key, classifiers)
        confusionMatrixPlot(cm, key, labels)
