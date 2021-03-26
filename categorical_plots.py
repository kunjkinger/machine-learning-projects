import seaborn as sns
%matplotlib inline 
import numpy as np

tips = sns.load_dataset('tips')

sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)
#estimator will tell here what is the std deviation of this category

sns.countplot(x='sex',data=tips)
#will give the count of variable 

sns.boxplot(x='day',y='total_bill',data=tips)
sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')
#hue will add another box plot for smokers here and the box split according to totalbill and smoker

sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',split=True) # it will show the kde 
 
sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex',split=True)

sns.swarmplot(x='day',y='total_bill',data=tips) # combination of strip and violinplot

# run together to make the combination of both above plots
sns.violinplot(x='day',y='total_bill',data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips,color='black')

#factorplot
sns.factorplot(x='day',y='total_bill',data=tips,kind='bar') 