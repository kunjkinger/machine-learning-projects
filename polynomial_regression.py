import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values #  in variable showing a matrix with 10,3 previously its only 10, which is a vector
y =dataset.iloc[:,2].values

#fitting the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =2) # degree is 2 because the polynomial eqn is b0 +b1x1 + b2(x2)^2
x_poly =  poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x) , color = 'blue')
plt.title('Truth or bluff(Linear model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# visualising the polynomial regression to the dataset
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)) , color = 'blue')
plt.title('Truth or bluff(polynomial model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# predict a new result with linear regression
lin_reg.predict([[6.5]])  # we cant use () this to predit here, because this expect to be 2d array that's y we use [[]]

# predict a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) # we use 6.5 because we want to predict the value on 6.5 among 10

