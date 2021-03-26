import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('College_Data',index_col=0)
df.head()
df.describe()

#scatterplot using seaborn
sns.lmplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private',fit_reg=False)

sns.lmplot(x='Outstate',y='F.Undergrad',data=df,hue='Private',fit_reg=False)


g = sns.FacetGrid(df,hue='Private',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


g = sns.FacetGrid(df,hue='Private',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


df[df['Grad.Rate']>100]

df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate']>100]


from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(df.drop('Private',axis=1))

km.cluster_centers_

def convert(Private):
    if Private == 'Yes':
        return 1
    else:
        return 0
    
df['cluster']= df['Private'].apply(convert)

df.head()

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['cluster'],km.labels_))
print('\n')
print(classification_report(df['cluster'],km.labels_))
