# Code to perform keyword extraction based on the RAKE algorithm. It outputs a CSV file
# containing a keyword set for each input paper
import re
from nltk.corpus import stopwords
import pandas as pd
import preprocessing
from rake_nltk import Rake

### MAIN CODE ###
TOP_K_KEYWORDS = 10 # top k number of keywords to retrieve in a ranked document

df = pd.read_csv('abstracts.csv')

text = preprocessing.preprocess(df['abstract'])

# Utilizing RAKE to get the keywords for each paper
rake = Rake(          # By default, it utilizes NLTK stop words
  punctuations = '',  # As the punctuation are removed in the preprocessing
  max_length = 3)     # from 1-gram to 3-grams keywords

df['text'] = text
result = []
for index, row in df.iterrows():
  rake.extract_keywords_from_text(row['text'])
  r = {}
  r['doi'] = row['doi']
  r['keywords'] = rake.get_ranked_phrases()[:TOP_K_KEYWORDS]
  result.append(r)

# Applyng rake to the union of text from all the papers
r = {}
r['doi'] = 'all papers'
rake.extract_keywords_from_text(text.str.cat(sep=' '))
r['keywords'] = rake.get_ranked_phrases()[:TOP_K_KEYWORDS]
result.append(r)

final = pd.DataFrame(result)
final.to_csv('extraction_rake.out.csv', index=False)