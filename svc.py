import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils import getData


def svc():
    X, y = getData()

    clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(X, y)

    return clf.score(X, y)


def gridSearch(X, y):

    paramGrid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001, 0.00001, 10]}

    clfGrid = GridSearchCV(SVC(), paramGrid, verbose=1, n_jobs=os.cpu_count())
    clfGrid.fit(X, y)

    print("Best Parameters:\n", clfGrid.best_params_)
    print("Best Estimators:\n", clfGrid.best_estimator_)
    return clfGrid


def gridSearchScore():
    X, y = getData()
    scores = ["precision", "recall"]
    for score in scores:
        print(f"# Tuning hyper-parameters for {score}")
        clf = gridSearch(X, y)

        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))


if __name__ == "__main__":
    gridSearchScore()