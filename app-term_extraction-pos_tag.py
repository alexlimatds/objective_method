from terms_extractors import PosTagExtractor, get_term_df
import data_readers

seed_corpus, seed_known_terms = data_readers.seed_set()
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

population_corpus, population_known_terms = data_readers.population_set()
population_occur_matrix = extractor.terms_by_doc(terms_idx, population_corpus, other_terms=population_known_terms)
population_term_df = get_term_df(population_occur_matrix)
with open("extracted_terms_df-population_set-pos_tag.csv", 'w') as csvfile:
  for i in range(len(terms)):
    csvfile.write(f'{i},{terms[i]},{population_term_df[i]}\n')

print(f'Done! Extracted {len(terms)} terms.')
