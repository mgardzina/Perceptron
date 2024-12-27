import math
import numpy as np


class Perceptron():
    def __init__(self):
        self.activation_function = lambda x: 1.5*math.tanh(x) + 1
        self.weights = []
        self.bias = 0

    def forward(self, x: list[float]) -> float:
        weighted_sum = np.dot(x, self.weights) + self.bias
        output = self.activation_function(weighted_sum)

        return output

    def train(self, x_train: list[list[float]], y_expected: list[float], n_iter: int, learning_rate: float):
        number_of_inputs = len(x_train[0])
        self.weights = np.random.rand(number_of_inputs)
        self.bias = np.random.rand()

        for _ in range(n_iter):
            for i, x in enumerate(x_train):
                y_predicted = self.forward(x)
                error = y_expected[i] - y_predicted
                correction = error * learning_rate

                self.weights += correction * x
                self.bias += correction

    def predict(self, x: list[list[float]]) -> list[float]:
        predictions = []

        for i, x in enumerate(x):
            output = self.forward(x)
            predictions.append(output)

        return predictions
