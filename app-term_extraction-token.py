import numpy as np
import data_readers
from terms_extractors import TokenExtractor, get_term_df

extractor = TokenExtractor()

# Extracting vocabulary
seed_text, seed_known_terms = data_readers.seed_set()
seed_corpus = data_readers.merge_corpus_and_terms(seed_text, seed_known_terms)
tokens, _ = extractor.extract_from_corpus(seed_corpus)
with open("extracted_terms-seed_set-token.csv", 'w') as csvfile:
  csvfile.write(f'{len(tokens)}\n')
  for i in range(len(tokens)):
    csvfile.write(f'{tokens[i]},{i}\n')

tokens_idx = {}
for i, t in enumerate(tokens):
  tokens_idx[t] = i

# Extracting tokens from the seed set
seed_occur_matrix = extractor.terms_by_doc(tokens_idx, seed_corpus)
seed_term_df = get_term_df(seed_occur_matrix)
with open("extracted_terms_df-seed_set-token.csv", 'w') as csvfile:
  for i in range(len(tokens)):
    csvfile.write(f'{i},{tokens[i]},{seed_term_df[i]}\n')
np.savetxt("occurrence_matrix-seed_set-token.csv", seed_occur_matrix)

# Extracting tokens from the population set
population_text, population_known_terms = data_readers.population_set()
population_corpus = data_readers.merge_corpus_and_terms(population_text, population_known_terms)
population_occur_matrix = extractor.terms_by_doc(tokens_idx, population_corpus)
population_term_df = get_term_df(population_occur_matrix)
with open("extracted_terms_df-population_set-token.csv", 'w') as csvfile:
  for i in range(len(tokens)):
    csvfile.write(f'{i},{tokens[i]},{population_term_df[i]}\n')

print(f'Done! Extracted {len(tokens)} terms.')
