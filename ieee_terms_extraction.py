from terms_extraction import PosTagExtractor
import pandas as pd

data = pd.read_csv('seed_set.csv')

extractor = PosTagExtractor()
corpus = data['title'].str.cat(data['abstract'], sep='. ', na_rep='').tolist()
known_terms = data['keywords'].apply(lambda x: x.split(',') if x is list else None).to_list()

terms, terms_df, occurrence_matrix = extractor.extract_with_df(corpus, other_terms=known_terms)

with open("ieee_extracted_terms-pos_tag.csv", 'w') as csvfile:
  csvfile.write('term,df\n')
  for i in range(len(terms)):
    csvfile.write(f'{terms[i]},{terms_df[i]}\n')
    
print(f'Done! Extracted {len(terms)} terms.')
