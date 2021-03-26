import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan_data.csv')
df.info()
df.describe()
 
#histogram with pandas using .hist()
plt.figure(figsize=(10,6))
df[df['credit.policy'] == 1]['fico'].hist(bins=35,
                                          label='credit policy = 1',
                                          alpha = 0.6)
df[df['credit.policy'] == 0]['fico'].hist(bins=35,
                                          label='credit_poicy = 0'
                                          ,alpha = 0.6)
plt.legend()
plt.xlabel('fico')


plt.figure(figsize=(10,6))
df[df['not.fully.paid'] == 1]['fico'].hist(bins=35,color='blue',
                                          label='not fully paid = 1',
                                          alpha = 0.6)
df[df['not.fully.paid'] == 0]['fico'].hist(bins=35,color='red',
                                          label='not fully paid = 0'
                                          ,alpha = 0.6)
plt.legend()
plt.xlabel('fico')

# countplot
plt.figure(figsize=(12,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')

#jointplot
plt.figure(figsize=(12,8))
sns.jointplot(x='fico',y='int.rate',data=df)

#lmplot
sns.lmplot(y='int.rate',x='fico',data=df,
           hue='credit.policy',col='not.fully.paid') 
#col used to seperate between columns in 0 and 1 



#categorical features making dummy variables
cat_feat = ['purpose']
final_data = pd.get_dummies(df,columns=cat_feat,drop_first=True)
final_data.info()

#train and test
x = final_data.drop('not.fully.paid',axis=1)
y= final_data['not.fully.paid']


#decision tree
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)

predictions = dtc.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))



#random forest
from sklearn.ensemble import RandomForestClassifier
rdf = RandomForestClassifier(n_estimators=300)
rdf.fit(x_train,y_train)
    
rdf_predictions = rdf.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,rdf_predictions))
print('\n')
print(classification_report(y_test,rdf_predictions))


#which one performs better
