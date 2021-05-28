"""
This module comprises the code related to the categorization and filtering of candidate terms.
"""
import random
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AffinityPropagationCategorizer:
  """
  This class uses the Affinity Propagation clustering algorithm to find the terms related 
  to a given category.
  """
  
  def __init__(self, term_vectors):
    """
    Creates an instance of this class. The clustering is performed by this constructor, so 
    the term vectors must be provided as argument.
    
    Parameters
    ----------
    term_vectors : numpy array of shape [number of terms, embedding dimension]
      The embedding vectors of the terms to be categorized.
    """
    self.term_vectors = term_vectors
    term_cosines = cosine_similarity(term_vectors)
    self.model = AffinityPropagation(affinity='precomputed', random_state=random.randint(0, 1000))
    self.model.fit(term_cosines)
  
  def number_of_clusters(self):
    """
    Returns
    -------
    int
      The number of clusters found by the clustering algorithm.
    """
    return self.model.cluster_centers_indices_.shape[0]
  
  def term_indices_for_category(self, category_vector):
    """
    Finds the terms related to a given category.
    
    Parameters
    ----------
    category_vector : numpy array of shape [embedding_dimension]
      The embedding vector of the category.
    
    Returns
    -------
    A numpy array of shape [n] where n is the number of found related terms.
      The indices of the terms related to a given category.
    """
    similarities = cosine_similarity(category_vector.reshape(1, -1), self.term_vectors)
    best_indices = np.argmax(similarities)                        # Finds the indices of the terms that are most similar to the category
    cluster_label = np.unique(self.model.labels_[best_indices])   # Get the labels of the clusters which the most similar terms were assigned
    if cluster_label.shape[0] > 1:                                # In an ideal case, the category is related to one cluster only. If this didn't happen, print a warning
      print('WARN [AffinityPropagationCategorizer]: I\'ve found more than one cluster for a category.')
    
    return np.nonzero(self.model.labels_ == cluster_label[0])[0]  # Returns the indices of the terms in the cluster of the category

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