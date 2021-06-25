import unittest
import numpy as np

from terms_vectorization import BertVectorizer
from terms_selector import TermSelector

class Test_TermSelector(unittest.TestCase):

  def setUp(self):
    self.dev_th = 0.2
    self.pop_th = 0.02
    self.vocabulary = {
      'software': 0, 'algorithm': 1, 'engineering': 2, 'cpu': 3, 'processing': 4, 
      'law': 5, 'legal': 6, 'dataset': 7, 'machine learning': 8, 'judgment': 9, 
      'court': 10, 'database': 11, 'neural network': 12, 'uml': 13, 'rice': 14, 
      'house': 15, 'dool': 16, 'data structure': 17, 'grass': 18, 'legislation': 19
    }
    # occurrence matrix of the candidate terms in the development set
    self.occurrence_matrix = np.array([
      [0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0], #doc 0: law,legal,dataset,machine learning,neural network
      [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0], #doc 1: legal,dataset,machine learning,
      [0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0], #doc 2: judgment,court,neural network
      [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0], #doc 3: legal,dataset
      [0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1]  #doc 4: law,machine learning,neural network,legislation
    ])
    # In the dev_df below, the values for the candidate terms follow the occurrence matrix.
    # Milesimal values, like .202, are used to have sure about the term sorting
    self.dev_df = [
      .15, .20, .11, .09, .30, 
      .40, .60, .60, .60, .202, 
      .201, .13, .60, .01, .01, 
      .08, .01, .11, .01, .20]
    self.pop_df = [
      .10, .15, .09, .08, .021, 
      .005, .006, .019, .02, .001,
      .001, .05, .018, .02, .0001,
      .004, .0001, .025, .0001, .001
    ]
    self.method = TermSelector(self.vocabulary, self.dev_df, self.dev_th, self.pop_df, self.pop_th)
    self.vectorizer = BertVectorizer()
  
  def test_init(self):
    candidates = self.method.candidates
    expected = [
      'law', 'legal', 'dataset', 'machine learning', 
      'judgment', 'court', 'neural network', 'legislation']
    
    for t in expected:
      self.assertTrue(t in candidates)
  
  def test_categorize_terms(self):
    self.method.categorize_terms(
      [['deep learning', 'neural network'], ['legal']], 
      self.vectorizer.get_embeddings)
    
    terms_by_category = self.method.terms_by_category
    self.assertTrue('law' in terms_by_category[1])
    self.assertTrue('legal' in terms_by_category[1])
    self.assertTrue('judgment' in terms_by_category[1])
    self.assertTrue('court' in terms_by_category[1])
    self.assertTrue('legislation' in terms_by_category[1])
    self.assertFalse('neural network' in terms_by_category[1])
    self.assertFalse('deep learning' in terms_by_category[1])

    self.assertTrue('dataset' in terms_by_category[0])
    self.assertTrue('machine learning' in terms_by_category[0])
    self.assertTrue('neural network' in terms_by_category[0])
    self.assertFalse('law' in terms_by_category[0])
    self.assertFalse('legal' in terms_by_category[0])
    self.assertFalse('judgment' in terms_by_category[0])
    self.assertFalse('court' in terms_by_category[0])

  def test_build_query_1(self):
    with self.assertRaises(RuntimeError):
      self.method.build_query(None)

  def test_build_query_2(self):
    self.method.categorize_terms(
      [['deep learning', 'neural network'], ['legal']], 
      self.vectorizer.get_embeddings)
    q_string = self.method.build_query(self.occurrence_matrix)
    q_terms = self.method.query_terms
    print(q_string)

    self.assertEqual(len(q_terms), 2)     # two categories
    self.assertEqual(len(q_terms[0]), 2)  # n of terms in the deep learning category
    self.assertEqual(len(q_terms[1]), 3)  # n of terms in the legal category
    self.assertTrue('dataset' in q_terms[0])
    self.assertTrue('neural network' in q_terms[0])
    self.assertTrue('legal' in q_terms[1])
    self.assertTrue('legislation' in q_terms[1])
    self.assertTrue('judgment' in q_terms[1] or 'court' in q_terms[1])

if __name__ == '__main__':
  unittest.main()