import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('ticks')
from terms_extraction import PosTagExtractor
from terms_vectorization import BertVectorizer
from terms_categorization import split_by_category
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

# Terms with df greater than one
sorted_df = np.argsort(term_df)

log += '** Extracted terms and their document frequency **\n'
for i in sorted_df[::-1]: # iterating from backwards
  if term_df[i] > 1:
    log += f'({term_df[i]} occurrences) {terms[i]} \n'

# Similarity

vectorizer = BertVectorizer()
term_vecs = vectorizer.get_embeddings(terms)
categories = ['deep learning', 'legal domain']
category_vecs = vectorizer.get_embeddings(categories)
'''
term_vecs = [[0.7, 0.2], [0.9, 0.1], [0.2, 0.7], [0.6, 0.3]] # a a b a
categories = ['a','b']
category_vecs = [[0.8, 0.1], [0.18, 0.9]]
'''
split = split_by_category(categories, category_vecs, term_vecs)

def sim(e):
  return e[1] # returns the cosine similarity value

log += '** Terms\' similarities grouped by closest category **\n'
for c in categories:
  log += f'{c}: {len(split[c])} terms\n'
  c_list = split[c]
  c_list.sort(reverse=True, key=sim)
  for t in c_list:
    log += f'\t{terms[t[0]]}: {t[1]}\n'

# Document Frequency vs Cossine similarity
_, axis = plt.subplots(len(categories), sharex=True, figsize=(10, 10))
for i, c in enumerate(categories):
  X, Y = [], []
  for t in split[c]:
    x = term_df[t[0]]
    X.append(x)
    y = t[1]
    Y.append(y)
    #axis[i].annotate(terms[t[0]], (x,y))
  axis[i].scatter(X, Y, s=3.0)
  axis[i].grid(axis='both')
  axis[i].set_title(c)

for ax in axis.flat:
  ax.set(xlabel='document frequency', ylabel='cossine similarity')

# Writing log file
log_file = open('analysis_extracted_terms-log.txt', 'w')
log_file.write(log)
log_file.close()

plt.show()
