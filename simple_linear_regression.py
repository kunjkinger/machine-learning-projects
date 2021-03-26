
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset= pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# selecting the data and train and test
from sklearn.model_selection import train_test_split
x_train,x_test, y_train , y_test = train_test_split(x,y,test_size = 1/3,random_state=0)

#fitting simple regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#predicting the test set results
y_pred = regressor.predict(x_test)

#visualising the training set results
plt.scatter(x_train , y_train , color='red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()  

#visualising the test set results
plt.scatter(x_test , y_test , color='red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue') # the regression line must be same otherwise it will build new regression line
plt.title('Salary Vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()  