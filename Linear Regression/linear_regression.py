import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("experience_salary_100.csv")

X = data["Experience_Years"].values
y = data["Salary_INR"].values

X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std


def hypothesis(thetha0, thetha1, X):
    return thetha0 + thetha1 * X


def loss_function(X, y, thetha0, thetha1):
    m = len(X)
    total_error = 0

    for i in range(m):
        prediction = hypothesis(thetha0, thetha1, X[i])
        error = prediction - y[i]
        total_error += error ** 2

    return total_error / (2 * m)


def gradient_descent(X, y, thetha0, thetha1, alpha, iterations):
    m = len(X)

    for _ in range(iterations):
        sum_error_0 = 0
        sum_error_1 = 0

        for i in range(m):
            prediction = hypothesis(thetha0, thetha1, X[i])
            error = prediction - y[i]
            sum_error_0 += error
            sum_error_1 += error * X[i]

        thetha0 = thetha0 - alpha * (sum_error_0 / m)
        thetha1 = thetha1 - alpha * (sum_error_1 / m)

    return thetha0, thetha1


def train_linear_regression(X, y, theta0, theta1, alpha, iterations):
    loss_history = []

    for _ in range(iterations):
        m = len(X)
        sum_error_0 = 0
        sum_error_1 = 0

        for i in range(m):
            prediction = hypothesis(theta0, theta1, X[i])
            error = prediction - y[i]
            sum_error_0 += error
            sum_error_1 += error * X[i]

        theta0 -= alpha * (sum_error_0 / m)
        theta1 -= alpha * (sum_error_1 / m)

        loss = loss_function(X, y, theta0, theta1)
        loss_history.append(loss)

    return theta0, theta1, loss_history


def predict(x, theta0, theta1):
    return theta0 + theta1 * x


theta0 = 0.0
theta1 = 0.0
alpha = 0.01
iterations = 2000

theta0, theta1, loss_history = train_linear_regression(
    X_scaled, y, theta0, theta1, alpha, iterations
)

print(theta0, theta1)

y_preds = [predict(x, theta0, theta1) for x in X_scaled]

idx = np.argsort(X)
X_sorted = X[idx]
y_preds_sorted = np.array(y_preds)[idx]

plt.scatter(X, y, s=20)
plt.plot(X_sorted, y_preds_sorted, color="red")
plt.grid(True)
plt.show()

plt.plot(loss_history)
plt.grid(True)
plt.show()
