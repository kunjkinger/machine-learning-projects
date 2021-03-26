import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')

sns.pairplot(iris,hue='species')
setosa = iris[iris['species']=='setosa']

sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True)

x = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)

pred = svm.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[.1,.01,.001,.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(x_train,y_train)

grid.best_estimator_
grid.best_params_


grid_pred = grid.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
