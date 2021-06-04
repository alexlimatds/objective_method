import unittest
import numpy as np
import terms_categorization

class TestAffinityPropagationCategorizer(unittest.TestCase):
  
  def test_model(self):
    term_vec = np.array([
      [0.98, 0.01, 0.02], [0.91, 0.11, 0.08], [0.81, 0.05, 0.10], 
      [0.18, 0.83, 0.07], [0.06, 0.88, 0.11], [0.03, 0.93, 0.07], 
      [0.11, 0.15, 0.99], [0.08, 0.09, 0.87], [0.09, 0.12, 0.83]])
    model = terms_categorization.AffinityPropagationCategorizer(term_vec)

    cat_vec = np.array([0.66, 0.31, 0.12])
    print(f'Number of clusters: {model.number_of_clusters()}')
    print(f'Vector indexes for category {cat_vec}: {model.term_indices_for_category(cat_vec)}')

    cat_vec = np.array([0.21, 0.14, 0.87])
    print(f'Number of clusters: {model.number_of_clusters()}')
    print(f'Vector indexes for category {cat_vec}: {model.term_indices_for_category(cat_vec)}')
    
class TestSplitByCategory(unittest.TestCase):
  
  def test__split_by_category__(self):
    categories = ['a','b']
    category_vectors = [[0.8, 0.1], [0.18, 0.9]]
    term_vectors = [[0.7, 0.2], [0.9, 0.1], [0.2, 0.7], [0.6, 0.3]] # a a b a
    
    dic = terms_categorization.split_by_category(categories, category_vectors, term_vectors)
    
    list_a = dic['a']
    self.assertEqual(list_a[0]['term_index'], 0)
    self.assertEqual(list_a[1]['term_index'], 1)
    self.assertEqual(list_a[2]['term_index'], 3)
    
    list_b = dic['b']
    self.assertEqual(list_b[0]['term_index'], 2)
    
if __name__ == '__main__':
  unittest.main()