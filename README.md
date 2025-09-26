# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

6.Define a function to predict the Regression value.

## Program and Output:
```
developed by : sanjay kumar B
reg no       : 212224230242
```
```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()
```
![433053515-a529740a-63e8-440e-9e99-2dbbf1274202](https://github.com/user-attachments/assets/2a70578e-a2bb-41b7-987e-d934c0c823e8)
```python
data = data.drop(['sl_no', 'salary'], axis=1)
data
```
![433053536-47d50c77-9ef9-4a1d-8845-04b6509e293d](https://github.com/user-attachments/assets/af347f31-6e2b-4edf-a60f-357a39b858fc)

```python
data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
```
![433053579-f50a1202-38da-4861-bf27-69a12bb462bd](https://github.com/user-attachments/assets/7ce2a409-8b62-4c72-b9d0-2b1b62fd80d1)
```python
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data
```
![433053599-650bf55c-7dcd-4e43-bdbb-b7df243f0efb](https://github.com/user-attachments/assets/7cb25b08-041e-4917-bd60-212610c97e4e)
```python
x = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 
y
```
![433053624-7d048205-f457-41e4-b12f-b8fb065040c9](https://github.com/user-attachments/assets/0e3fb59c-bab4-4ef2-9f31-37f50805dee7)
```python
theta = np.random.randn(x.shape[1]) 
Y = y
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y): 
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations): 
    m = len(y)
    for i in range(num_iterations): 
        h = sigmoid(X.dot(theta)) 
        gradient = X.T.dot(h - y) / m 
        theta -= alpha * gradient 
    return theta
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred

y_pred = predict(theta, x) 
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy) 
print(y_pred)
```
![433053660-f8752f06-1332-4b30-aa93-ccd4e9868aae](https://github.com/user-attachments/assets/bf4560eb-fba5-4cad-ab24-203156005520)
```python
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
```
![433053676-04d4f991-705b-49db-ac00-61c026ef77ba](https://github.com/user-attachments/assets/dd89e227-3983-4599-bbe0-4d565e7d0c21)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
