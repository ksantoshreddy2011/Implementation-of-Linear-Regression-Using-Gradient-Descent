# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start

2.Initialize data, parameters, and learning rate

3.Repeat for given iterations: • Predict output • Compute loss • Update weight and bias

4.Plot results and display parameters

5.Stop

## Programme
```

/*
Program to implement the linear regression using gradient descent.
Developed by: SANTHOSH REDDY K
RegisterNumber: 212225240137
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values


x = (x - np.mean(x)) / np.std(x)

w = 0.0          
b = 0.0          
alpha = 0.01     
epochs = 100
n = len(x)

losses = []
for i in range(epochs):
   y_hat = w * x + b
   loss = np.mean((y_hat - y) ** 2)
   losses.append(loss)
   dw = (2/n) * np.sum((y_hat - y) * x)
   db = (2/n) * np.sum(y_hat - y)
   w = w - alpha * dw
   b = b - alpha * db

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line",color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()
print("Final Weight (w):", w)
print("Final Bias (b):", b)
```

## Output:
![linear regression using gradient descent]<img width="1370" height="576" alt="MLEXP3" src="https://github.com/user-attachments/assets/f4689d2d-3f27-43b8-a04e-4d302c0b235f" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
