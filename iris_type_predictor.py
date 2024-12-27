import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from perceptron import Perceptron

# Load the iris dataset and extract the features and labels from it to train the perceptron
iris_dataset = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.3, random_state=5)

perceptron = Perceptron()
perceptron.train(x_train, y_train, n_iter=30, learning_rate=0.02)

y_predicted = perceptron.predict(x_test)
y_predicted = [math.ceil(y) for y in y_predicted]
accuracy = accuracy_score(y_test, y_predicted)

print(f"Accuracy: {round(accuracy,3)*100}%")
