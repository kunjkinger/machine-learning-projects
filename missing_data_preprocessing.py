import numpy as np # to include any kind of mathematics
import matplotlib.pyplot as plt #
import pandas as pd # for import and manage the dataset


dataset = pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,strategy  = 'mean')

imputer=imputer.fit(x[:, 1:3])
x[:,1:3]=imputer.transform(x[:, 1:3])
