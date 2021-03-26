import seaborn as sns
%matplotlib inline

tips = sns.load_dataset('tips')

sns.lmplot('total_bill','tip',tips,hue='sex',markers=['o','v'],scatter_kws={'s':50})
#scatter_kws not used generally and s tends for size
sns.lmplot('total_bill','tip',tips,col='sex',row='time') # col for seperating sex in differnt maps unlike hue

sns.lmplot('total_bill','tip',tips,col='day',hue='sex',aspect=0.6,size=8) 