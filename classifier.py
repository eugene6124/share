#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import argparse
import datetime
import pickle as pkl
from utils import getDataLoader, getPredictDataLoader, toNumpy
from monitor import EarlyStopping, matrix
from connector import DB

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class Network(nn.Module):
    def __init__(self, D_in, D_out):
        super(Network, self).__init__()
        self.D_in = D_in
        self.fc1 = nn.Linear(D_in, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc31 = nn.Linear(25, 25)
        self.fc32 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 10)
        self.fc5 = nn.Linear(10, D_out)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc31.weight)
        torch.nn.init.xavier_uniform_(self.fc32.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x = x.view(-1, self.D_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc32(x))
        # x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class Predictor:
    def __init__(self, args):

        self.params = args

        (self.trainLoader, self.testLoader, self.D_in, self.D_out) = getDataLoader(
            trainRatio=self.params.trainRatio,
            batchSize=self.params.batchSize,
            accidentType=self.params.accidentType,
        )
        self.model = Network(self.D_in, self.D_out).to(DEVICE)

        self.opt = optim.Adam(
            self.model.parameters(), lr=self.params.learningRate, weight_decay=1e-5
        )
        self._criterion = torch.nn.CrossEntropyLoss()

    def criterion(self, output, target):
        if self.D_out == 1:
            return self._criterion(output, target)
        else:
            return self._criterion(output, torch.max(target, 1)[1])

    def train(self):
        self.model.train()
        losses = []

        for batchIdx, (data, target) in enumerate(self.trainLoader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = self.model(data)
            self.opt.zero_grad()

            loss = self.criterion(output, target)
            loss.backward()
            self.opt.step()

            losses.append(toNumpy(DEVICE, loss))

        return sum(losses) / len(losses)

    def evaluate(self):
        self.model.eval()
        losses = []

        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in self.testLoader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)

                total += data.size(0)

                _check = toNumpy(DEVICE, torch.max(target, 1)[1]) == toNumpy(
                    DEVICE, torch.max(output, 1)[1]
                )
                correct += _check.sum()

                if losses:
                    losses.append(
                        toNumpy(
                            DEVICE,
                            self.criterion(output, target),
                        )
                    )
                else:
                    losses = [
                        toNumpy(
                            DEVICE,
                            self.criterion(output, target),
                        )
                    ]

        print("Accuracy for {} : {:.2f}%".format(total, correct / total * 100))
        return sum(losses) / len(losses)

    def predict(self):
        self.model.eval()

        predictLoader, searchKey = getPredictDataLoader(
            batchSize=self.params.batchSize,
            accidentType=self.params.accidentType,
        )

        predictList = []

        with torch.no_grad():
            for data in predictLoader:
                data = data.to(DEVICE)
                output = self.model(data)

                if predictList:
                    predictList.append(
                        toNumpy(DEVICE, torch.max(output, 1)[1]).reshape(-1, 1)
                    )
                else:
                    predictList = [
                        toNumpy(DEVICE, torch.max(output, 1)[1]).reshape(-1, 1)
                    ]

        predictResult = np.concatenate(tuple(predictList), axis=0).reshape(-1, 1)
        docs = pd.DataFrame(
            np.concatenate(
                (searchKey, predictResult),
                axis=1,
            ),
            columns=["link_id", "time_code", "dfk_code", "accident_ratio"],
        ).to_dict("records")

        return docs

    def loadPredict(self):
        self.model.load_state_dict(torch.load("checkpoint.pt"))
        response = DB().query("create", "road_accident", self.predict())
        print(response)

    def fit(self):
        self.earlyStopping = EarlyStopping(patience=self.params.patience, verbose=True)

        endEpoch = self.params.epochs
        for epoch in range(1, self.params.epochs + 1):
            train_loss = self.train()
            print("[{}] Train Loss : {:.6f}".format(epoch, train_loss))
            test_loss = self.evaluate()
            print("[{}] Test  Loss : {:.6f}".format(epoch, test_loss))
            self.earlyStopping(test_loss, self.model)
            if self.earlyStopping.early_stop:
                endEpoch = epoch
                break

        logList = [
            self.differ(*row)
            for loader in [self.trainLoader, self.testLoader]
            for row in loader
        ]

        logList = np.concatenate(np.array(logList), axis=0)

        resultMatrix = matrix(logList)

        print(
            f"Early Stopping {endEpoch}\nCollect Ratio : {resultMatrix.values[-1, -1]:.6f}\n{logList}"
        )

        # with open(
        #     "data/result_{}.pkl".format(
        #         datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        #     ),
        #     "wb",
        # ) as f:
        #     pkl.dump(resultMatrix, f)
        # print(resultMatrix)

        response = DB().query("create", "road_accident", self.predict())
        print(response)

        return resultMatrix.values[-1, -1]

    def differ(self, data, target):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = self.model(data)
        modelResult = np.array(
            list(
                zip(
                    toNumpy(DEVICE, torch.max(target, 1)[1].view(-1)),
                    toNumpy(DEVICE, torch.max(output, 1)[1].view(-1)),
                )
            )
        )
        return modelResult


def classifier():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learningRate", "-lr", type=float, default=1e-4)
    parser.add_argument("--trainRatio", "-t", type=float, default=0.8)

    parser.add_argument("--batchSize", "-b", type=int, default=4096)
    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--patience", "-p", type=int, default=20)
    parser.add_argument("--accidentType", "-at", type=int, default=0)

    args = parser.parse_args()
    CollectRatioList = []
    return Predictor(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learningRate", "-lr", type=float, default=1e-4)
    parser.add_argument("--trainRatio", "-t", type=float, default=0.8)

    parser.add_argument("--batchSize", "-b", type=int, default=4096)
    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--patience", "-p", type=int, default=20)
    parser.add_argument("--accidentType", "-at", type=int, default=0)

    args = parser.parse_args()
    CollectRatioList = []
    Predictor(args).loadPredict()
    print(CollectRatioList)