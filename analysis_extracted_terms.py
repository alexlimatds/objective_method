import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mticker
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('ticks')
from terms_extraction import PosTagExtractor
from terms_vectorization import BertVectorizer
from terms_categorization import split_by_category
import analysis_functions
import numpy as np

data = pd.read_csv('seed_set.csv')
corpus = data['title'].str.cat(data[['abstract', 'keywords']], sep='. ', na_rep='').tolist()

extractor = PosTagExtractor()
terms, term_df, occur_matrix = extractor.extract_with_df(corpus)
log = ''

# Histogram of all extracted terms
max_df = int(max(term_df))
hist = np.bincount(term_df)
bins = np.arange(max_df + 1)

_, hist_plt = plt.subplots(figsize=(7, 5))
hist_plt.set_title('Term DF histogram')
hist_plt.bar(bins, hist)
hist_plt.grid(axis='y')
hist_plt.set_yscale('log')
hist_plt.set(xlabel='Term DF', ylabel='Frequency')
hist_plt.set_xticks(bins)
hist_plt.text(max_df - 3, max(hist), f'{len(terms)} terms in total')

plt.savefig('analysis_extracted_terms_histogram.pdf', bbox_inches='tight')

# Terms with df greater than one
sorted_df = np.argsort(term_df)

log += '** Extracted terms with document frequency above one **\n'
for i in sorted_df[::-1]: # iterating from backwards
  if term_df[i] > 1:
    log += f'({term_df[i]} occurrences/DF) {terms[i]} \n'

# Similarity

vectorizer = BertVectorizer()
term_vecs = vectorizer.get_embeddings(terms)
categories = ['deep learning', 'legal domain']
category_vecs = vectorizer.get_embeddings(categories)
'''
# test data
term_vecs = [[0.7, 0.2], [0.9, 0.1], [0.2, 0.7], [0.6, 0.3]] # a a b a
categories = ['a','b']
category_vecs = [[0.8, 0.1], [0.18, 0.9]]
'''
split = split_by_category(categories, category_vecs, term_vecs)

log += '** Terms\' similarities and DF grouped by closest category **\n'
for c in categories:
  log += f'{c}: {len(split[c])} terms\n'
  c_array = split[c]
  sorted_array = np.sort(c_array, order=['similarity'])[::-1]
  for t in sorted_array:
    log += f'\t{terms[t[0]]}: {t[1]} - {term_df[t[0]]} occurrences/DF\n'

# Document Frequency vs Cossine similarity
for i, c in enumerate(categories):
  _, axis = plt.subplots(1, figsize=(20, 20))
  X, Y = [], []
  for t in split[c]:
    x = term_df[t['term_index']] # df
    X.append(x)
    y = t['similarity']          # similarity
    Y.append(y)
    if x >= 4 or y >= 0.8:
      axis.annotate(terms[t['term_index']], (x,y))
  axis.scatter(X, Y, s=3.0)
  axis.set_xticks(range(max_df + 2))
  label_format = '{:,.1f}'
  y_ticks = np.arange(0.0, 1.2, 0.1).tolist()
  axis.set_yticks(y_ticks)
  axis.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
  axis.set_yticklabels([label_format.format(x) for x in y_ticks])
  axis.grid(axis='both')
  axis.set_title(c)
  axis.set(xlabel='document frequency', ylabel='cossine similarity')

  plt.savefig(f'analysis_extracted_terms_df_vs_sim-{c}.pdf', bbox_inches='tight')

# TODO histogram of extracted terms by doc

# Map of occurrence matrix
for i, c in enumerate(categories):
  c_array = split[c]
  c_occur_matrix = occur_matrix[:, c_array['term_index']] # occurrence matrix including just the terms of the current category
  sims = split[c]
  
  # filtering
  f = c_array['similarity'] >= 0.5
  c_occur_matrix = c_occur_matrix[:, f]
  sims = sims[f]
  
  analysis_functions.plot_map(
    c_occur_matrix, 
    terms=terms, 
    similarities=sims, 
    fig_size=(100,20), 
    save_name=f'analysis_extracted_terms_occurrence_matrix-{c}.pdf')
  
# Writing log file
log_file = open('analysis_extracted_terms-log.txt', 'w')
log_file.write(log)
log_file.close()

#plt.show()