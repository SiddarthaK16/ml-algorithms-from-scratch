import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("lasso_test_data.csv")

X = data.drop("target", axis=1).values   
y = data["target"].values

#
scaler = StandardScaler()
X = scaler.fit_transform(X)

m, n = X.shape



def hypothesis(theta0, theta, X_row):
    return theta0 + np.dot(theta, X_row)  


def cost_function(X, y, theta0, theta, lamb):
    m = len(X)
    total_error = 0

    for i in range(m):
        prediction = hypothesis(theta0, theta, X[i])
        error = prediction - y[i]
        total_error += error ** 2

    l1_penalty = np.sum(np.abs(theta))

    return total_error / (2*m) + (lamb / (2*m)) * l1_penalty



def gradient_descent(X, y, theta0, theta, alpha, lamb, iterations):
    m, n = X.shape

    for _ in range(iterations):
        sum_error_0 = 0
        sum_error = np.zeros(n)

        for i in range(m):
            prediction = theta0 + np.dot(theta, X[i])
            error = prediction - y[i]

            sum_error_0 += error
            sum_error += error * X[i]

        sum_error_0 = sum_error_0 / m

         
        lasso_term = np.sign(theta)

        sum_error = (sum_error / m) + (lamb / m) * lasso_term

        
        theta0 = theta0 - alpha * sum_error_0
        theta = theta - alpha * sum_error

    return theta0, theta



def train_lasso(X, y, theta0, theta, alpha, iterations, lamb):
    return gradient_descent(X, y, theta0, theta, alpha, lamb, iterations)



def predict(theta0, theta, X):
    return theta0 + np.dot(X, theta)






theta0 = 0
theta = np.zeros(n)

alpha = 0.01
iterations = 1000
lamb = 1


theta0, theta = train_lasso(X, y, theta0, theta, alpha, iterations, lamb)


y_pred = predict(theta0, theta, X)


plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Lasso)")

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.savefig("lasso_actual_vs_predicted.jpg", dpi=300)
plt.show()


residuals = y - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.savefig("lasso_residuals.jpg", dpi=300)
plt.show()


plt.figure()
plt.bar(range(len(theta)), theta)
plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.title("Lasso Feature Weights")

plt.savefig("lasso_feature_weights.jpg", dpi=300)
plt.show()