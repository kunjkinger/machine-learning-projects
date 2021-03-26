import seaborn as sns
%matplotlib inline

tips = sns.load_dataset('tips')
#dist plot is for only 1 variable 
sns.distplot(tips['total_bill'],kde=False) # kde just shows the histogram
# kde mean kearnel density estimation 

#this tells us the comparison between 2 variables
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg') # if we provide kind that it will change the 
#structure like here print regression line otherwise it will show only scatter plot
#try kind with hex 
sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
#kde will show 2d and show where the intensity is high

# joint plot use with default scatter
# it will show every plots 
sns.pairplot(data=tips)
sns.pairplot(tips,hue='sex') # it will give the comparison between 2 variables

sns.rugplot(tips['total_bill']) # it will show you how many data points liw in that area like 
# in here it shows around the data ponts between 10 and 20

sns.kdeplot(tips['total_bill'])