import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nltk

yelp = pd.read_csv('yelp.csv')

yelp.info()
yelp.describe()

yelp['textlength'] = yelp['text'].apply(len)

sns.set_style('white')

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'textlength')

sns.boxplot(x='stars',y='textlength',data=yelp)

sns.countplot(x='stars',data=yelp)

stars = yelp.groupby('stars').mean()

stars.corr()

plt.figure(figsize=(12,8))
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

#nlp classification task
yelp_class = yelp[(yelp['stars']==1)| (yelp['stars']==5)]
yelp_class

x = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x = cv.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train,y_train)

predictions = nb.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


#using text processing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow',CountVectorizer()),
                  ('tfidf',TfidfTransformer()),
                  ('model',MultinomialNB())])

x = yelp_class['text']
y = yelp_class['stars']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

pipe.fit(x_train,y_train)

pred = pipe.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
