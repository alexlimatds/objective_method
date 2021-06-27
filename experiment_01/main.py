import pandas as pd
import numpy as np

import data_readers
from terms_selector import TermSelector
from terms_vectorization import BertVectorizer

vocabulary = data_readers.read_vocabulary("extracted_terms-seed_set-token.csv")

dev_set = pd.read_csv('seed_set.csv')
len_dev_set = dev_set.shape[0]
dev_df = data_readers.read_df("extracted_terms_df-seed_set-token.csv", vocabulary, n_docs=len_dev_set)
dev_occur_matrix = np.loadtxt("occurrence_matrix-seed_set-token.csv")

population_set = pd.read_csv('population_set_ieee.csv')
len_pop_set = population_set.shape[0]
pof_df = data_readers.read_df("extracted_terms_df-population_set-token.csv", vocabulary, n_docs=len_pop_set)

category_terms = [
  ["deep learning", "neural network"], 
  ["legal", "law"]]
term_vectorizer = BertVectorizer()

# TODO grid search
selector = TermSelector(vocabulary, dev_df, 0.2, pof_df, 0.02)
selector.categorize_terms(category_terms, term_vectorizer.get_embeddings)
query = selector.build_query(dev_occur_matrix)

print(f'Done! Resulting query: {query}')