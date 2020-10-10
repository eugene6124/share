import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from sklearn.model_selection import train_test_split


class BinaryClassifier(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = nn.Linear(D_in, D_out)
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class Logistic:
    def __init__(self):
        (self.trainX, self.trainY), (self.testX, self.testY) = self.getData()

        self.trainX = torch.from_numpy(self.trainX).float()
        self.trainY = torch.from_numpy(self.trainY.reshape(-1, 1)).float()
        self.testX = torch.from_numpy(self.testX).float()
        self.testY = torch.from_numpy(self.testY.reshape(-1, 1)).float()

        self.model = BinaryClassifier(self.trainX.shape[1], 1)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-2)

        self.nb_epochs = 1000

    def getData(self):
        X, y = utils.getData()

        trainX, testX, trainY, testY = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=1234
        )

        return (trainX, trainY), (testX, testY)

    def train(self):
        hypothesis = self.model(self.trainX)

        cost = F.binary_cross_entropy(hypothesis, self.trainY)

        self.opt.zero_grad()
        cost.backward()
        self.opt.step()

        return cost

    def evaluation(self):
        hypothesis = self.model(self.testX)
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == self.testY
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        return accuracy

    def fit(self):
        print("Logic")
        for epoch in range(self.nb_epochs + 1):
            # 100번마다 로그 출력
            if epoch % 100 == 0:
                print(
                    "Epoch {:4d}/{} Accuracy: {:.6f}".format(
                        epoch, self.nb_epochs, self.evaluation() * 100
                    )
                )


if __name__ == "__main__":
    Logistic().fit()