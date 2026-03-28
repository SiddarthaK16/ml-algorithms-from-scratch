import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("ridge_regression_dataset_1_feature.csv")

X= data["x"].values
y=data["y"].values

scalar=StandardScaler()
X = data["x"].values.reshape(-1, 1)
X = scalar.fit_transform(X)
X = X.flatten()   # convert back to 1D   

def hypothesis (thetha0 , thetha1 , X):
    return thetha0 + (thetha1*X)

def cost_function(X,y,thetha0,thetha1,lamb):   #lamb=lambda
    m=len(X)
    total_error=0

    for i in range(m):
        prediction = hypothesis(thetha0 , thetha1 , X[i])
        error= prediction - y[i]
        total_error+=error ** 2

    return total_error / (2*m) + (lamb / (2*m)) * (thetha1**2)


def gradient_descent(X,y,thetha0 , thetha1 , alpha , lamb , iterations):
    m=len(X)
    

    for _ in range(iterations):
        sum_error_0=0
        sum_error_1=0
    

        for i in range(m):
            prediction = thetha0 + (thetha1 * X[i])
            error = prediction - y[i]

            sum_error_0+=error
            sum_error_1+= error * X[i]

        sum_error_0 = sum_error_0 / m 
        sum_error_1 = (sum_error_1 / m) + (lamb / m) * thetha1 #ridge term

        thetha0 = thetha0 - alpha * (sum_error_0)
        thetha1 = thetha1 - alpha * (sum_error_1)

    return thetha0 , thetha1


def train_ridge_regression (X,y,thetha0,thetha1 , alpha , iterations , lamb):
    
    
    thetha0 , thetha1 = gradient_descent(X,y,thetha0,thetha1 , alpha , lamb , iterations)

    return thetha0,thetha1

def predict(thetha0 , thetha1 , X):
    return thetha0 + (thetha1*X)

def r2_score(y,y_preds):
    ss_total= ((y - y.mean())**2).sum()
    ss_residual = ((y-y_preds)**2).sum()

    return 1 - (ss_residual/ss_total)



thetha0 = 0
thetha1 = 0
alpha=0.01
iterations = 1000
lamb=1

thetha0 , thetha1 = train_ridge_regression (X,y,thetha0,thetha1 , alpha , iterations , lamb)

y_pred = predict ( thetha0 , thetha1 , X)

import matplotlib.pyplot as plt
import numpy as np



# sort for smooth line
sorted_idx = np.argsort(X)
X_sorted = X[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure()
plt.scatter(X, y)
plt.plot(X_sorted, y_pred_sorted)
plt.title("Ridge Regression Fit")
plt.xlabel("X")
plt.ylabel("y")

plt.savefig("regression_fit.jpg", dpi=300)
plt.show()



plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.savefig("actual_vs_predicted.jpg", dpi=300)
plt.show()

print (thetha0 , thetha1)






                                     





    
