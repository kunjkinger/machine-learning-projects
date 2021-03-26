#multiple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets= pd.read_csv('50_Startups.csv')
x = datasets.iloc[:,:-1].values
y=datasets.iloc[:,4].values


#categorical data encoding the independent vatiable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,-1]=labelencoder_x.fit_transform(x[:,-1])
onehotencoder =OneHotEncoder(categorical_features=[-1])
x = onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x = x[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

#fitting linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predict the set test results
y_pred = regressor.predict(x_test)

# building the optimal model using backward elimination
import statsmodels.api as sm
x =np.append(arr= np.ones((50,1)).astype(int),values =x , axis = 1)
 # 50 because there are 50 var in x ,1 because of 1 coloumn we wwant to add bcz f eqn b0 + b1x1
 
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()