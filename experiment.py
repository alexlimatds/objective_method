import pandas as pd
import numpy as np
import datetime

import data_readers
from terms_selector import TermSelector
from terms_vectorization import BertVectorizer

class Experiment:
  """
  Use an instance of this class to perform an set of experiments. Each experiment applies different 
  values of term document frequency thresholds (grid search), but all of them utilizes the same data files.
  """

  def __init__(self, v_path, dev_set_path, pop_set_path, df_dev_path, df_pop_path, 
  dev_occur_matrix_path, category_terms, dev_thresholds, pop_thresholds):
    """
    Stores the experiments' parameters and loads the related data.

    Parameters
    ----------
    v_path : string
      The path of the vocabulary file.
    dev_set_path : string
      The path of the development set file.
    pop_set_path : string
      The path of the population set file.
    df_dev_path : string
      The path of the file containing the term document frequencies regarding the development set.
    df_pop_path : string
      The path of the file containing the term document frequencies regarding the population set.
    dev_occur_matrix_path : string
      The path of the file containing the term-document occurrence matrix regarding the development set.
      The file must follows the numpy format.
    category_terms : list of list of str
      The terms related to the desired categories. Use a sublist for each category.
    dev_thresholds : list of float
      The thresholds to be applied in order to select terms when regarding the term document frequencies 
      from the development set (first filter of terms).
    pop_thresholds : list of float
      The thresholds to be applied in order to select terms when regarding the term document frequencies 
      from the population set (second filter of terms).
    """
    # Saving paths
    self.vocabulary_path = v_path
    self.dev_set_path = dev_set_path
    self.pop_set_path = pop_set_path
    self.df_dev_path = df_dev_path
    self.df_pop_path = df_pop_path
    self.dev_occur_matrix_path = dev_occur_matrix_path
    # Loading data
    self.vocabulary = data_readers.read_vocabulary(v_path)
    dev_set_len = pd.read_csv(dev_set_path).shape[0]
    pop_set_len = pd.read_csv(pop_set_path).shape[0]
    self.dev_df = data_readers.read_df(df_dev_path, self.vocabulary, n_docs=dev_set_len)
    self.pop_df = data_readers.read_df(df_pop_path, self.vocabulary, n_docs=pop_set_len)
    self.dev_occur_matrix = np.loadtxt(dev_occur_matrix_path)
    # Saving parameters
    self.category_terms = category_terms
    self.dev_thresholds = dev_thresholds
    self.pop_thresholds = pop_thresholds

    self.term_vectorizer = BertVectorizer()
  
  def run(self):
    """
    Performs the experiments.

    Returns
    -------
    string : The experiments' report.
    """
    self.timestamp = datetime.datetime.now()
    self.report = (
      "Experiment Report\n"
      "-----------------\n"
      f"{self.timestamp.strftime('%Y/%m/%d - %H:%M:%S')}\n"
      f"Vocabulary path:      {self.vocabulary_path}\n"
      f"Development set path: {self.dev_set_path}\n"
      f"Population set path:  {self.pop_set_path}\n"
      f"Development DF path:  {self.df_dev_path}\n"
      f"Population DF path:   {self.df_pop_path}\n"
      f"Development occurrence matrix path: {self.dev_occur_matrix_path}\n"
      )
    for dev_th in self.dev_thresholds:
      for pop_th in self.pop_thresholds:
        selector = TermSelector(self.vocabulary, self.dev_df, dev_th, self.pop_df, pop_th)
        selector.categorize_terms(self.category_terms, self.term_vectorizer.get_embeddings)
        q = selector.build_query(self.dev_occur_matrix)
        q_array = selector.query_array
        coverture = np.count_nonzero(q_array)
        doc_count = q_array.shape[0]
        self.report += (
          f"\nDevelopment threshold: {dev_th}\n"
          f"Population threshold: {pop_th}\n"
          f"Generated query: {q}\n"
          f"Development set coverture: {coverture / doc_count * 100.0:.2f}% ({coverture} out of {doc_count} documents) \n"
          )
    return self.report
