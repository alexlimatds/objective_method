import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('white')

out_str = ''

### Author keywords' analysis

df = pd.read_csv('seed_set.csv')

# keywords' histogram
k_hist = {}
for index, row in df[df['keywords'].notna()].iterrows():
  l = row['keywords'].lower().split(',')
  for k in l:
    k = k.strip()
    k_hist[k] = k_hist.get(k, 0) + 1

df2 = pd.DataFrame({'keyword': k_hist.keys(), 'frequency': k_hist.values()}).sort_values(by='keyword', ascending=False)
ax = df2.plot(
  kind='barh', 
  x='keyword', 
  y='frequency',
	figsize=(7, 12));
ax.set_ylabel('Author Keywords')
ax.set_xlabel('Frequency')
plt.savefig('seed_set_analysis-author_keywords.pdf', bbox_inches='tight')

