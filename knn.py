import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def knn(classes):
    X, y, _ = utils.getData(samplingType="1", classes=classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123456
    )

    classifier = KNeighborsClassifier(n_neighbors=classes, n_jobs=os.cpu_count())
    print("start")
    classifier.fit(X_train, y_train)
    print("end")

    # X_test, y_test, _ = utils.getData(samplingType="none", classes=classes)
    y_pred = classifier.predict(X_test)
    print("predict")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    knn(2)