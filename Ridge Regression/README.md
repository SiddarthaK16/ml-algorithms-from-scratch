# Ridge Regression (From Scratch)

This is my implementation of Ridge Regression from scratch using Python.

I didn’t use any ML libraries for training — everything like the cost function, gradient calculation, and parameter updates is written manually to understand how it works internally.

What’s implemented:

* Gradient Descent
* L2 Regularization (Ridge)
* Feature Scaling
* Prediction function
* Basic plots

Output:
I plotted the regression line and actual vs predicted values. The line looks almost perfectly straight.

Why it looks too perfect:
I did not use a train-test split. The model is trained and tested on the same dataset, so it already “knows” the answers and fits the data extremely well.

This project is mainly for understanding the math and building the algorithm from scratch, not for perfect evaluation.

Tech used:
Python, NumPy, Matplotlib , Scikit-Learn (only for scaling)

Goal:
To actually understand ML by implementing it instead of just using libraries.

