#!/usr/bin/env python
# coding: utf-8

import pickle as pkl
import numpy as np
import pandas as pd
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from elasticsearch import Elasticsearch, helpers


class MyDataLoader(Dataset):
    def __init__(self, accidentType=0):
        X, y, columnX = getData(accidentType, "under")

        self.X = from_numpy(X).float()
        self.y = from_numpy(oneHot(y).values).long()  # .float()`
        print(self.X.shape, self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def getDim(self):
        return self.X.shape[1], self.y.shape[1]


class PredictDataLoader(Dataset):
    def __init__(self, accidentType=0):
        X, y, columnX, searchKey = getPredictData(accidentType)
        self.X = from_numpy(X).float()
        self.searchKey = searchKey

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

    def getDim(self):
        return self.X.shape[1]

    def getSearchKey(self):
        return self.searchKey


def getRowData(accidentType=0, windows=1, classes=2):
    columnX = [
        "LINK_ID",
        "year",
        "occrrnc_time_code",
        "dfk_code",
        "rn",
        "ROAD_RANK",
        "MAX_SPD",
        "LANES",
        "F_LINES",
        "F_LANES",
        "T_LINES",
        "T_LANES",
        "degree",
        "SLOPE",
        "ANGLE",
    ]
    columnY = ["acdnt_gae2_cnt", "dprs2_cnt", "sep2_cnt", "slp2_cnt", "inj_aplcnt2_cnt"]
    es = Elasticsearch("121.162.20.187:9200", timeout=60)
    rawData = pd.DataFrame.from_dict(
        [
            doc["_source"]
            for doc in helpers.scan(
                es,
                index="dataset",
                size=10000,
            )
        ]
    )

    print(len(rawData))

    x = rawData[columnX]
    y = rawData[columnY[accidentType]]

    x["year"] = pd.to_numeric(x["year"])

    if classes == 2:
        y[y > 0] = 1
    else:
        y[(y > 0) & (y < 4)] = 1
        y[y > 3] = 2

    dataset = x.copy()
    dataset["y"] = y.values
    return addPastAccident(dataset, windows=windows)


def getData(accidentType=0, samplingType="auto", windows=1, classes=2):
    dataset = getRowData(accidentType, windows, classes)
    dataset = dataset[dataset["year"] > 2016]
    dataset.drop(columns=["LINK_ID", "year"], inplace=True)

    scaler, dataset = normalization(dataset)

    assert samplingType in ["over", "under", "auto", "none", "1"]

    if samplingType == "auto":
        dataset = pd.concat(list(autoSampling(dataset)))
    elif samplingType != "none":
        zero = dataset[dataset["y"] == 0]
        nonZero = dataset[dataset["y"] != 0]

        if samplingType == "over":
            dataset = pd.concat([sampling(nonZero, zero.shape[0]), zero])
            assert zero.shape[0] == dataset.shape[0] // 2
        if samplingType == "under":
            dataset = pd.concat([sampling(zero, nonZero.shape[0]), nonZero])
            assert nonZero.shape[0] == dataset.shape[0] // 2
        if samplingType == "1":
            dataset = pd.concat(
                list(autoSampling(dataset, length=dataset[dataset["y"] == 1].shape[0]))
            )

    x = dataset[dataset.columns.difference(["y"])]
    y = dataset["y"]
    print(x.columns)
    print(dataset["y"].value_counts())

    return x.values, y.values, x.columns


def getPredictData(accidentType=0, window=1, classes=2):
    dataset = getRowData(accidentType, window, classes)
    dataset["FMEAN"] = dataset["FLANES"] / dataset["FLINES"]
    dataset["TMEAN"] = dataset["TLANES"] / dataset["TLINES"]
    dataset = dataset[dataset["year"] == np.max(dataset["year"].values)]
    dataset.drop(columns=["year"], inplace=True)

    x = dataset[dataset.columns.difference(["y", "LINK_ID"])]
    _, x = normalization(x)
    y = dataset["y"]

    searchKey = ["LINK_ID", "occrrnc_time_code", "dfk_code"]
    print(x.shape, dataset[searchKey].shape)
    return (
        x.values,
        y.values,
        x.columns,
        dataset[searchKey].values,
    )


def addPastAccident(dataset, windows=0):
    for i in range(1, windows + 1):
        uniqueKey = ["LINK_ID", "occrrnc_time_code", "year"]
        columns = ["LINK_ID", "occrrnc_time_code", "year", "y"]
        windowDataset = pd.DataFrame(dataset[columns].values, columns=columns)
        windowDataset.rename(columns={"y": f"y_{i}"}, inplace=True)
        windowDataset[f"y_{i}"] = pd.to_numeric(windowDataset[f"y_{i}"])
        windowDataset["year"] = windowDataset["year"].apply(lambda y: y - 1)
        dataset = pd.merge(windowDataset, dataset, on=uniqueKey)
    return dataset


def autoSampling(dataset, length=None):
    labels = dataset["y"].unique().shape[0]
    size = dataset["y"].shape[0] / labels / 2
    for y in dataset["y"].unique().tolist():
        target = dataset[dataset["y"] == y]
        if length:
            sampleSize = int(length + target.shape[0]) // 2
            for _ in range(2):
                sampleSize = int(sampleSize + length) // 2
            yield sampling(target, sampleSize)
        else:
            sampleSize = int(size + (target.shape[0] / 2)) // 10
            yield sampling(target, sampleSize)


def normalization(dataset):
    columnRename = {c: c.replace("_", "") for c in dataset.columns}
    dataset.rename(columns=columnRename, inplace=True)
    print(dataset.columns)
    print(dataset.dtypes)

    dataset = oneHot(dataset)

    continuousColumns = [c for c in dataset.columns if "_" not in c]
    categoricalColumns = [c for c in dataset.columns if "_" in c]

    standardScaler = StandardScaler()
    standardScaler.fit(dataset[categoricalColumns].values)

    return standardScaler, pd.DataFrame(
        np.concatenate(
            [
                dataset[continuousColumns].values,
                standardScaler.transform(dataset[categoricalColumns].values),
            ],
            axis=1,
        ),
        columns=continuousColumns + categoricalColumns,
    )


def oneHot(dataset):
    return pd.get_dummies(dataset)


def sampling(df, n_samples):
    isReplace = True if df.shape[0] < n_samples else False
    return resample(df, replace=isReplace, n_samples=n_samples, random_state=1234)


def getDataLoader(trainRatio=0.7, batchSize=128, accidentType=0):
    datas = MyDataLoader(accidentType)
    Din, Dout = datas.getDim()
    trainSize = round(len(datas) * trainRatio)
    valSize = len(datas) - trainSize
    trainDataSet, valDataSet = random_split(datas, [trainSize, valSize])
    trainSampler = RandomSampler(trainDataSet)
    trainLoader = DataLoader(
        trainDataSet,
        batch_size=batchSize,
        shuffle=False,
        sampler=trainSampler,
        drop_last=True,
    )
    valSampler = RandomSampler(valDataSet)
    valLoader = DataLoader(
        valDataSet,
        batch_size=batchSize,
        shuffle=False,
        sampler=valSampler,
        drop_last=True,
    )
    return trainLoader, valLoader, Din, Dout


def getPredictDataLoader(batchSize=128, accidentType=0):
    datas = PredictDataLoader(accidentType)
    predictLoader = DataLoader(
        datas,
        batch_size=batchSize,
        shuffle=False,
        drop_last=False,
    )
    return predictLoader, datas.getSearchKey()


def toNumpy(DEVICE, tensor):
    if DEVICE.type == "cuda":
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


if __name__ == "__main__":
    # for i in range(5):
    x, y = getData(accidentType=0)
    # print(x, y)
    print(y.shape)
    print(y[0])
