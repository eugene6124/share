import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.decomposition import PCA
import utils


def getNComponents(X, columnX):
    df = pd.DataFrame(X, columns=columnX)

    pca = PCA()
    pca.fit(df)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    exp_var_sub = [
        round(a - b, 10) for a, b in zip(exp_var_cumul[1:], exp_var_cumul[:-1])
    ]

    index = 0
    for v in exp_var_sub:
        if v > 0.05:
            index += 1
        else:
            break
    return df, index + 1


def getFeature(X, columnX):
    df, index = getNComponents(X, columnX)
    pca = PCA(n_components=index)
    return index, pca.fit_transform(df).round(6)


def corr(X, y, x_columns):
    data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    columns = x_columns.values.tolist()
    columns.append("y")

    corrMatrix = pd.DataFrame(data, columns=columns).corr()

    corrResult = sorted(
        list(zip(columns[:-1], corrMatrix.iloc[-1].values.tolist()[:-1])),
        key=lambda x: -x[1],
    )

    setCorr(result, matrix)

    return corrResult, corrMatrix


def setCorr(result, matrix):
    with open("data/corr.pkl", "wb") as f:
        pkl.dump([result, matrix], f)


def getCorr():
    with open("data/corr.pkl", "rb") as f:
        result, matrix = pkl.load(f)
    return result, matrix


def pca():
    X, y, x_columns = utils.getData(samplingType="none")
    result, matrix = corr(X, y, x_columns)
    n_columns, feature = getFeature(X, x_columns)


if __name__ == "__main__":
    pca()