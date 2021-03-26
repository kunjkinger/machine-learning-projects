import numpy as np # to include any kind of mathematics
import matplotlib.pyplot as plt #
import pandas as pd # for import and manage the dataset

# importing the dataset
dataset = pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

# find out the missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,strategy  = 'mean')

imputer=imputer.fit(x[:, 1:3])
x[:,1:3]=imputer.transform(x[:, 1:3])

#encoding the categorical data
# categorical to numerical value
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x= onehotencoder.fit_transform(x).toarray()
labelencoder_y= LabelEncoder()
y= labelencoder_y.fit_transform(y)

# splitting the dataset in to the training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)