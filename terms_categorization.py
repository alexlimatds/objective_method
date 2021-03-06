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
  to a given category term.
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
    Finds the terms related to a given category term.
    
    Parameters
    ----------
    category_vector : numpy array of shape [embedding_dimension]
      The embedding vector of the category term.
    
    Returns
    -------
    A numpy array of shape [n] where n is the number of found related terms.
      The indices of the terms related to a given category term.
    """
    similarities = cosine_similarity(category_vector.reshape(1, -1), self.term_vectors)
    best_indices = np.argmax(similarities)                        # Finds the indices of the terms that are most similar to the category
    cluster_label = np.unique(self.model.labels_[best_indices])   # Get the labels of the clusters which the most similar terms were assigned
    if cluster_label.shape[0] > 1:                                # In an ideal case, the category is related to one cluster only. If this didn't happen, print a warning
      print('WARN [AffinityPropagationCategorizer]: I\'ve found more than one cluster for a category.')
    
    return np.nonzero(self.model.labels_ == cluster_label[0])[0]  # Returns the indices of the terms in the cluster of the category

def split_by_category(categories, category_vectors, term_vectors):
  """
  Splits the terms according to the similarity among them and the categories.
  
  Parameters
  ----------
  categories : list of string
    The category terms.
  category_vectors : numpy array of shape (number of categories, embedding dimension)
    The respective embedding vectors of the category terms.
  term_vectors : numpy array of shape (number of terms, embedding dimension)
    The embedding vectors of the terms.
    
  Returns
  -------
    A dictionary containing one entry for each category, where the category term is the key. The 
    entry value is a numpy structured array, from which each record holds the term_index 
    and similarity (cosine similarity) fields.
  """
  similarities = cosine_similarity(term_vectors, category_vectors)
  cat_terms = {}
  for i, cat in enumerate(categories):
    terms_idx = np.nonzero(np.argmax(similarities, axis=1) == i)[0]
    tuples = []
    for idx in terms_idx:
      tuples.append((idx, similarities[idx, i]))
    cat_terms[cat] = np.array(tuples, dtype=[('term_index', int), ('similarity', float)])
  
  return cat_terms
