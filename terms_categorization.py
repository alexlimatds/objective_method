"""
This module comprises the code related to the categorization and filtering of candidate terms.
"""
import random
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AffinityPropagationCategorizer:
  """
  TODO
  """
  
  def __init__(self, term_vectors):
    self.term_vectors = term_vectors
    term_cosines = cosine_similarity(term_vectors)
    self.model = AffinityPropagation(affinity='precomputed', random_state=random.randint(0, 1000))
    self.model.fit(term_cosines)
  
  def number_of_clusters(self):
    return self.model.cluster_centers_indices_.shape[0]
  
  def term_indices_for_category(self, category_vector):
    similarities = cosine_similarity(category_vector.reshape(1, -1), self.term_vectors)
    best_indices = np.argmax(similarities)
    cluster_label = np.unique(self.model.labels_[best_indices])
    if cluster_label.shape[0] > 1:
      print('WARN [AffinityPropagationCategorizer]: I\'ve found more than one cluster for a category.')
    
    return np.nonzero(self.model.labels_ == cluster_label[0])

# Test
def test_AffinityPropagationCategorizer():
  term_vec = np.array([
    [0.98, 0.01, 0.02], [0.91, 0.11, 0.08], [0.81, 0.05, 0.10], 
    [0.18, 0.83, 0.07], [0.06, 0.88, 0.11], [0.03, 0.93, 0.07], 
    [0.11, 0.15, 0.99], [0.08, 0.09, 0.87], [0.09, 0.12, 0.83]])
  model = AffinityPropagationCategorizer(term_vec)
  
  cat_vec = np.array([0.66, 0.31, 0.12])
  print(f'Number of clusters: {model.number_of_clusters()}')
  print(f'Vector indexes for category {cat_vec}: {model.term_indices_for_category(cat_vec)}')
  
  cat_vec = np.array([0.21, 0.14, 0.87])
  print(f'Number of clusters: {model.number_of_clusters()}')
  print(f'Vector indexes for category {cat_vec}: {model.term_indices_for_category(cat_vec)}')