import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %matplotlib inline

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

import utils

import pickle as pkl


def rf(classes=3):
    X, y, _ = utils.getData(samplingType="1", classes=classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123456
    )

    rf = RandomForestClassifier(
        n_estimators=10,
        oob_score=True,
        random_state=123456,
        n_jobs=os.cpu_count(),
        criterion="entropy",
    )
    rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print(f"Out-of-bag score estimate: {rf.oob_score_:.3}")

    cm = pd.DataFrame(
        confusion_matrix(y_test, predicted),
    )
    sns.heatmap(cm, annot=True)

    with open(f"./result/rf_result_{classes}.pkl", "wb") as f:
        pkl.dump(
            [
                ["model", "predict", "accuracy", "out-of-bag", "cm"],
                [rf, predicted, accuracy, rf.oob_score_, cm],
            ],
            f,
        )

    print(f"Mean accuracy score: ", accuracy_score(y_test, predicted))
    print("precision: ", metrics.precision_score(y_test, predicted))
    print("recall: ", metrics.recall_score(y_test, predicted))
    print("f1: ", metrics.f1_score(y_test, predicted))
    print(classification_report(y_test, predicted))

    return y_test, predicted


def load(classes=3):
    X, y, _ = utils.getData(samplingType="1", classes=classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123456
    )

    rf, predicted, accuracy, oob_score, cm = pkl.load(
        open(f"./result/rf_result_{classes}.pkl", "rb")
    )[1]

    predicted = rf.predict(X_test)

    print(cm)

    print(f"Mean accuracy score: ", accuracy_score(y_test, predicted))
    print("precision: ", metrics.precision_score(y_test, predicted))
    print("recall: ", metrics.recall_score(y_test, predicted))
    print("f1: ", metrics.f1_score(y_test, predicted))
    print(classification_report(y_test, predicted))

    return y_test, predicted


def result(dataLoad=True):
    classes = 2
    if dataLoad:
        y_test, predicted = load(classes)
    else:
        y_test, predicted = rf(classes)


if __name__ == "__main__":
    result(False)
    # result(True)
