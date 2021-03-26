import seaborn as sns 

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
tips.head()

tc = tips.corr()
sns.heatmap(tc,annot=True,cmap='coolwarm') # annot will give the numbers on the map too in the block

fp = flights.pivot_table(index='month',columns='year',values='passengers')
sns.heatmap(fp,cmap='magma',linecolor='white',linewidths=3) #heatmap describes the most flights easily using color

sns.clustermap(fp,cmap='coolwarm',standard_scale=1)