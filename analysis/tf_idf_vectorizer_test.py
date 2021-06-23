import unittest
import terms_extractors
from analysis.tf_idf_vectorizer import TfidfVectorizer

class TfidfVectorizerTest(unittest.TestCase):
  def test_(self):
    corpus = [
      'My name is John.', 
      'My name is Bond, James Bond.',
      "I don't have a nickname", 
      "What's your nick?", 
      "Please, sign your name in this field.", 
      "We applied word embeddings as text features and inputs to the neural network."
    ]
    keywords = [None, None, ['nick'], ['nick'], None, ['deep learning']]

    extractor = terms_extractors.PosTagExtractor()
    pipe = TfidfVectorizer(extractor)
    pipe.fit(corpus, keywords)
    matrix = pipe.__count_terms__(corpus, keywords)
    v = pipe.vocabulary_dic_

    self.assertEqual(matrix[2,v['nick']], 1)          # counting for nick in doc 2
    self.assertEqual(matrix[3,v['nick']], 2)          # counting for nick in doc 3
    self.assertEqual(matrix[5,v['deep learning']], 1) # counting for deep learning in doc 5

if __name__ == '__main__':
  unittest.main()