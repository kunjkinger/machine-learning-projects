import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('covid_19_data.csv')
x=dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test results
y_pred = regressor.predict(x_test)

#fitting the poynomial 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly =poly_reg.fit_transform(x_test)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y_test,sample_weight= None)

#visualisng the train set
plt.scatter(x_train,y_train,color ='red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('chances of recover patients from total')
plt.xlabel('total cases')
plt.ylabel('recovered cases')
plt.show()
#visualisng the test set
plt.scatter(x_test,y_test,color ='red')
plt.plot(x_test,regressor.predict(x_test),color = 'blue')
plt.title('chances of recover patients from total')
plt.xlabel('total cases')
plt.ylabel('recovered cases')
plt.show()

# visualising the polynomial set
plt.scatter(x_test,y_test,color ='red')
plt.plot(x_test,lin_reg.predict(poly_reg.fit_transform(x_test)),color = 'blue')
plt.title('chances of recover patients from total')
plt.xlabel('total cases')
plt.ylabel('recovered cases')
plt.show()