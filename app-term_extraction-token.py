from numpy import append
import data_readers
from terms_extractors import TokenExtractor, get_term_df

def merge(text, known_terms):
  corpus = []
  for i in range(len(seed_text)):
    if known_terms[i]:
      corpus.append(text[i] + " " + ", ".join(known_terms[i]))
    else:
      corpus.append(text[i])
  return corpus

extractor = TokenExtractor()

seed_text, seed_known_terms = data_readers.seed_set()
seed_corpus = merge(seed_text, seed_known_terms)
terms, _ = extractor.extract_from_corpus(seed_corpus)
with open("extracted_terms-seed_set-token.csv", 'w') as csvfile:
  csvfile.write(f'{len(terms)}\n')
  for i in range(len(terms)):
    csvfile.write(f'{terms[i]},{i}\n')

terms_idx = {}
for i, t in enumerate(terms):
  terms_idx[t] = i

seed_occur_matrix = extractor.terms_by_doc(terms_idx, seed_corpus)
seed_term_df = get_term_df(seed_occur_matrix)
with open("extracted_terms_df-seed_set-token.csv", 'w') as csvfile:
  for i in range(len(terms)):
    csvfile.write(f'{i},{terms[i]},{seed_term_df[i]}\n')

population_text, population_known_terms = data_readers.population_set()
population_corpus = merge(population_text, population_known_terms)
population_occur_matrix = extractor.terms_by_doc(terms_idx, population_corpus)
population_term_df = get_term_df(population_occur_matrix)
with open("extracted_terms_df-population_set-token.csv", 'w') as csvfile:
  for i in range(len(terms)):
    csvfile.write(f'{i},{terms[i]},{population_term_df[i]}\n')

print(f'Done! Extracted {len(terms)} terms.')
