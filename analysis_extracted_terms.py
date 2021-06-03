import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('white')
from terms_extraction import PosTagExtractor
import numpy as np

data = pd.read_csv('seed_set.csv')
corpus = data['title'].str.cat(data[['abstract', 'keywords']], sep='. ', na_rep='').tolist()

extractor = PosTagExtractor()
terms, term_df, occur_matrix = extractor.extract_with_df(corpus)

# Histogram of all extracted terms
max_df = int(max(term_df))
hist = np.bincount(term_df)
bins = np.arange(max_df + 1)
print(hist)
print(bins)

_, hist_plt = plt.subplots(figsize=(7, 5))
hist_plt.set_title('Term DF histogram')
hist_plt.bar(bins, hist)
hist_plt.grid(axis='y')
hist_plt.set_yscale('log')
hist_plt.set(xlabel='Term DF', ylabel='Frequency')
hist_plt.set_xticks(bins)
hist_plt.text(max_df - 3, max(hist), f'{len(terms)} terms in total')

# Terms with df greater than one
sorted_df = np.argsort(term_df)
log = ''
for i in sorted_df[::-1]: # iterating from backwards
  if term_df[i] > 1:
    log += f'({term_df[i]} occurrences) {terms[i]} \n'

log_file = open('analysis_extracted_terms-log.txt', 'w')
log_file.write(log)
log_file.close()

    
plt.show()
