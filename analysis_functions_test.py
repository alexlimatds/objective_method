import unittest
import analysis_functions
import numpy as np

class Test_plot_map(unittest.TestCase):
  
  def test_1(self):
    occur_matrix = np.array([
      [1, 1, 1, 0, 0, 1, 0, 1, 1], 
      [0, 1, 1, 1, 1, 0, 0, 0, 0], 
      [0, 0, 0, 1, 1, 1, 0, 1, 1], 
      [1, 1, 0, 1, 0, 1, 1, 0, 1]])
    terms = ['term 1', 'term 2', 'term 3', 'term 4', 'term 5', 'term 6', 'term 7', 'term  8', 'term 9']
    similarities = np.array([
      (0, 0.8), (1, 0.7), (2, 0.3), (3, 0.4), (4, 0.55), (5, 0.6), (6, 0.2), (7, 0.75), (8, 0.92001)],
      dtype=[('term_index', int), ('similarity', float)])

    analysis_functions.plot_map(
      occur_matrix, 
      terms=terms, 
      similarities=similarities, 
      show=True, 
      save_name=None, 
      fig_size=(5,5))

if __name__ == '__main__':
  unittest.main()