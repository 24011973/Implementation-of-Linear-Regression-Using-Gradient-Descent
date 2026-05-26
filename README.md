# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Start with an initial guess for the slope m and intercept b (usually zero).
2. Predict and Calculate Error: Compute the predicted values y(hat) = mx + b and find the difference from the actual values y.
3. Compute Gradients: Calculate the partial derivatives (direction of steepest ascent) for both m and b based on the Mean Squared Error.
4. Update Weights: Adjust m and b by moving in the opposite direction of the gradient by a factor of the learning rate (alpha).
   

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ETHICVARAN S
RegisterNumber:  212224230072
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Startup.csv")

X = df['R&D Spend'].values
y = df['Profit'].values

X = (X - X.mean()) / X.std()

m = 0
b = 0

lr = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = m * X + b
    
    # 
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - lr * dm
    b = b - lr * db

print("Slope (m):", m)
print("Intercept (b):", b)

y_pred = m * X + b

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
```

## Output:
<img width="750" height="562" alt="image" src="https://github.com/user-attachments/assets/d4efcb67-52f3-4e65-8a99-c56dc95a070b" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
