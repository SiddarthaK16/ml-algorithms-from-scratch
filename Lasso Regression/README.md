# Lasso Regression (From Scratch)

This is a simple implementation of **Lasso Regression** using Python and NumPy, built from scratch without using machine learning libraries.

---

## What is Lasso Regression?

Lasso Regression is a linear regression technique that adds a **penalty on the absolute values of the coefficients (L1 regularization)**.

This helps:

* Reduce overfitting
* Automatically remove useless features
* Keep the model simple and interpretable

---


## How it works

* Initialize weights and bias
* Predict using a linear function
* Compute error
* Update weights using gradient descent
* Apply L1 penalty using the **sign function**
* Repeat for multiple iterations

---

## Dataset

A synthetic dataset is used with:

* 10 features (x0 to x9)
* Only first 3 features actually affect the output
* Remaining features are just noise

---

## Expected Result

After training:

* Important features → non-zero weights
* Useless features → weights shrink to **0**

This shows Lasso performing **feature selection**.

---

## Visualizations

The model generates:

* Actual vs Predicted plot
* Residual plot
* Feature weights graph

These help understand performance and which features matter.

---

## Final Note

Lasso is not just about fitting data —
it’s about **deciding which features deserve to stay**.

That’s what makes it powerful.

