from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

"""
This class defines an object capable to build a query string from a vocabulary and 
the related data as term document freqeuncy and term document occurrence.
"""
class TermSelector:
  
  def __init__(self, vocabulary, dev_df, dev_threshold, pop_df, pop_threshold):
    """
    Creates an instance of this class and performs the selection of the candidate 
    terms. In order to get a query, call the categorize_terms and build_query methods, 
    in that order, after create the instance with this constructor. The selection of the 
    candidate is based on the provided thresholds and document frequencies.

    Parameters
    ----------
    vocabulary : a dictionary
      This dictionary must map each term (key) to its index (value).
    dev_df : a list of float values
      The document frequency values of the vocabulary's terms according to the
      development set. The indexes follow the vocabulary's term index.
    dev_threshold : a float value
      The development set threshold applied during the first term selection, which 
      will select the terms with a dev_df >= dev_threshold.
    pop_df : a list of float values
      The document frequency values of the vocabulary's terms according to the
      population set. The indexes follow the vocabulary's term index.
    pop_threshold : a float value
      The population set threshold applied during the second term selection, which 
      will select the terms (selected by the first filter) with a pop_df <= pop_threshold.
    """
    self.vocabulary = vocabulary
    self.dev_df = dev_df
    self.pop_df = pop_df
    self.dev_th = dev_threshold
    self.pop_th = pop_threshold
    self.terms_by_category = None

    # Selecting candidate terms
    self.candidates = []
    for term, idx in vocabulary.items():
      if self.dev_df[idx] >= self.dev_th and self.pop_df[idx] <= pop_threshold:
        self.candidates.append(term)
  
  def categorize_terms(self, category_terms, term_vectorizer):
    """
    Splits the candidate terms in category sets. For each category, you must provide one or
    more category terms. For each candidate term it is computed the cosine similarities among it 
    and the category terms. Then, the candidate term is assigned to the category of the most similar 
    category term. In order to compute the cosine similarities, you must provide a function 
    to vectorize the terms (strings) into numpy arrays.

    Parameters
    ----------
    category_terms : list of list of string
      The category terms. Each sublist represents a category.
    term_vectorizer : a callable
      A function having a list of string as parameter and that returns a numpy array of shape 
      (n, m), where n is the number of elements in the input list, and m is the dimension of 
      the embedding vectors.
    """
    n_categories = len(category_terms)
    n_candidates = len(self.candidates)
    # Vectorizing the category terms
    self.category_vectors = [None] * n_categories
    for i, c_list in enumerate(category_terms):
      v = term_vectorizer(c_list) # v is a matrix with one line per category term
      self.category_vectors[i] = v
    
    # Vectorizing candidate terms and assigning them into categories
    self.terms_by_category = [[] for i in range(n_categories)]
    v_candidates = term_vectorizer(self.candidates) # v_candidates is a matrix with one line per candidate term
    best_sims = np.zeros((n_candidates, n_categories))
    for i, cat_matrix in enumerate(self.category_vectors):
      sims = cosine_similarity(v_candidates, cat_matrix) # sims.shape: (n_candidates, n of terms in the category)
      best_sims[:,i] = np.amax(sims, axis=1)
    best_categories = np.argmax(best_sims, axis=1) # best_categories.shape: (n_candidates, 1)
    for i in range(n_candidates):
      cat_idx = best_categories[i]
      self.terms_by_category[cat_idx].append(self.candidates[i])

    # Removing categories without terms
    temp = []
    for l in self.terms_by_category:
      if len(l) > 0:
        temp.append(l)
    self.terms_by_category = temp

  def build_query(self, occurrence_matrix):
    """
    Builds the query, i.e., selects the candidate terms that will compose the query string.
    The categorize_terms method must be called before or an exception will be raised. 
    After calling this method, the selected terms and the query string will be respectively 
    available through the query_terms and the query attibutes. The selection is based on the 
    occurrence of the candidate terms over a set of documents, usually the development set or 
    the validation set.

    Parameters
    ----------
    occurrence_matrix : a numpy array of shape (n_docs, n_terms)
      A binary matrix indicating the occurrence (1 value) or the absence (0 value) of 
      each term in the vocabulary over a set of documents. Each line represents a 
      document and each column represents a term.

    Returns
    -------
    The query string
    """
    if not self.terms_by_category:
      raise RuntimeError("The categorize_terms method was not previously called.")
    
    # Creating binary arrays of categories
    n_categories = len(self.terms_by_category)
    q_array = self.__query_array__(self.terms_by_category, occurrence_matrix)
    
    # Sorting the terms by their development DF
    def sort_by_df(term):
      return self.dev_df[self.vocabulary[term]]
    sorted_cat_terms = []
    for cat_list in self.terms_by_category:
      sorted_list = cat_list.copy()
      sorted_list.sort(key=sort_by_df, reverse=True)
      sorted_cat_terms.append(sorted_list)

    # Searching and removing the dismissible terms
    query_terms = self.__copy_2d_list__(sorted_cat_terms)
    for cat_idx, t_list in enumerate(sorted_cat_terms):
      for term in t_list:
        k_terms_by_category = self.__copy_2d_list__(query_terms)
        k_terms_by_category[cat_idx].remove(term)
        k_v = self.__query_array__(k_terms_by_category, occurrence_matrix)
        k_q = np.logical_and(q_array, k_v)
        if np.array_equal(q_array, k_q):
          query_terms[cat_idx].remove(term)
    self.query_terms = query_terms
    self.query_array = self.__query_array__(query_terms, occurrence_matrix)

    # Creating string representation of the final query
    self.query = '(' + ') AND ('.join([' OR '.join(t) for t in self.query_terms]) + ')'

    return self.query

  def __query_array__(self, terms_by_category, occurrence_matrix):
    # building binary category arrays
    n_categories = len(terms_by_category)
    n_docs = occurrence_matrix.shape[0]
    arrays = np.zeros((n_categories, n_docs))
    for cat_idx, t_list in enumerate(terms_by_category):
      for term in t_list:
        term_idx = self.vocabulary[term]
        term_occurrences = occurrence_matrix[:,term_idx]
        arrays[cat_idx] += term_occurrences
      arrays[cat_idx] = (arrays[cat_idx] > 0)
    
    # building query array    
    q_array = arrays[0]
    for i in range(1, n_categories):
      q_array = np.logical_and(q_array, arrays[i])
    
    return q_array

  def __copy_2d_list__(self, list):
    return [sublist.copy() for sublist in list]
