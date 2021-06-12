from terms_extraction import PosTagExtractor, get_term_df
import pandas as pd

data = pd.read_csv('seed_set.csv')
seed_corpus = data['title'].str.cat(data['abstract'], sep='. ', na_rep='').tolist()
seed_known_terms = data['keywords'].apply(lambda x: x.split(',') if x is list else None).to_list()

extractor = PosTagExtractor()
terms, _ = extractor.extract_from_corpus(seed_corpus, other_terms=seed_known_terms)
with open("extracted_terms-seed_set-pos_tag.csv", 'w') as csvfile:
  csvfile.write(f'{len(terms)}\n')
  for i in range(len(terms)):
    csvfile.write(f'{terms[i]},{i}\n')

terms_idx = {}
for i, t in enumerate(terms):
  terms_idx[t] = i

seed_occur_matrix = extractor.terms_by_doc(terms_idx, seed_corpus, other_terms=seed_known_terms)
seed_term_df = get_term_df(seed_occur_matrix)
with open("extracted_terms_df-seed_set-pos_tag.csv", 'w') as csvfile:
  for i in range(len(terms)):
    csvfile.write(f'{i},{terms[i]},{seed_term_df[i]}\n')

data = pd.read_csv('population_set_ieee.csv')
population_corpus = data['title'].str.cat(data['abstract'], sep='. ', na_rep='').tolist()
population_known_terms = data['terms'].apply(lambda x: x.split(',') if x is list else None).to_list()

population_occur_matrix = extractor.terms_by_doc(terms_idx, population_corpus, other_terms=population_known_terms)
population_term_df = get_term_df(population_occur_matrix)
with open("extracted_terms_df-population_set-pos_tag.csv", 'w') as csvfile:
  for i in range(len(terms)):
    csvfile.write(f'{i},{terms[i]},{population_term_df[i]}\n')

print(f'Done! Extracted {len(terms)} terms.')
