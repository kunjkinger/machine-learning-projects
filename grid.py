import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

iris = sns.load_dataset('iris')

sns.pairplot(iris) 
g = sns.PairGrid(iris) # pairgrid we can control it according to us rather than pairplot
#g.map(plt.scatter)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

tips = sns.load_dataset('tips')
g = sns.FacetGrid(tips,col='time',row='smoker')
#g.map(sns.distplot,'total_bill')
g.map(plt.scatter,'total_bill','tip')