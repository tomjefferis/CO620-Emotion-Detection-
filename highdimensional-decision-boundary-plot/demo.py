import uci_loader
import numpy as np
from decisionboundaryplot import DBPlot
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from numpy.random.mtrand import permutation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    # load data
    X, y = uci_loader.getdataset("iris")
    # data = load_digits(n_class = 2)
    # X,y = data.data, data.target
    y[y != 0] = 1

    # shuffle data
    random_idx = permutation(np.arange(len(y)))
    X = X[random_idx]
    y = y[random_idx]

    # create model
    model = LogisticRegression(C=1)
    # model = RandomForestClassifier(n_estimators=10)

    # plot high-dimensional decision boundary
    db = DBPlot(model)
    db.fit(X, y, training_indices=0.5)
    db.plot(
        plt, generate_testpoints=True
    )  # set generate_testpoints=False to speed up plotting
    
    plt.show()

    # plot learning curves for comparison
    N = 10
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    plt.errorbar(
        train_sizes,
        np.mean(train_scores, axis=1),
        np.std(train_scores, axis=1) / np.sqrt(N),
    )
    plt.errorbar(
        train_sizes,
        np.mean(test_scores, axis=1),
        np.std(test_scores, axis=1) / np.sqrt(N),
        c="r",
    )

    plt.legend(["Accuracies on training set", "Accuracies on test set"])
    plt.xlabel("Number of data points")
    plt.title(str(model))
    plt.show()
