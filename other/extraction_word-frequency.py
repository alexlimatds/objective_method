# Code to perform keyword extraction based on word frequency. It outputs a CSV file
# containing a keyword set for each input paper and a keyword set for all the papers
import re
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import preprocessing

# Functions from https://www.kaggle.com/rowhitswami/keywords-extraction-using-tf-idf-method

def sort_coo(coo_matrix):
  """Sort a dict with highest score"""
  tuples = zip(coo_matrix.col, coo_matrix.data)
  return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
  """get the feature names and tf-idf score of top n items"""

  #use only topn items from vector
  sorted_items = sorted_items[:topn]

  score_vals = []
  feature_vals = []

  # word index and corresponding tf-idf score
  for idx, score in sorted_items:

      #keep track of feature name and its corresponding score
      score_vals.append(round(score, 3))
      feature_vals.append(feature_names[idx])

  #create a tuples of feature, score
  results= {}
  for idx in range(len(feature_vals)):
      results[feature_vals[idx]]=score_vals[idx]

  return results

def get_keywords(vectorizer, feature_names, doc):
  """Return top k keywords from a doc using TF-IDF method"""
  #generate tf-idf for the given document
  tf_idf_vector = vectorizer.transform([doc])
  
  #sort the tf-idf vectors by descending order of scores
  sorted_items=sort_coo(tf_idf_vector.tocoo())
  
  #extract only TOP_K_KEYWORDS
  keywords=extract_topn_from_vector(feature_names,sorted_items,TOP_K_KEYWORDS)
  
  return list(keywords.keys())

### MAIN CODE ###
TOP_K_KEYWORDS = 10 # top k number of keywords to retrieve in a ranked document

df = pd.read_csv('abstracts.csv')

text = preprocessing.preprocess(df['abstract'])

vectorizer = CountVectorizer(
  stop_words=stopwords.words('english'), 
  ngram_range=(1, 3))
vectorizer.fit_transform(text)
feature_names = vectorizer.get_feature_names()

# Adding the prevailing terms in each paper
df['text'] = text
result = []
for index, row in df.iterrows():
  r = {}
  r['doi'] = row['doi']
  r['keywords'] = get_keywords(vectorizer, feature_names, row['text'])
  result.append(r)

# Adding the prevailing terms for all the papers
r = {}
r['doi'] = 'all papers'
r['keywords'] = get_keywords(vectorizer, feature_names, text.str.cat(sep=' '))
result.append(r)

final = pd.DataFrame(result)
final.to_csv('extraction_word-frequency.out.csv', index=False)