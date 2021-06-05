import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mticker
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('ticks')
import numpy as np

def plot_map(matrix, terms=None, similarities=None, show=False, vertical_grid=True, save_name=None, fig_size=(30,20)):
  '''
  Plot of a matrix of documents (matrix lines) vs terms' occurrences (matrix columns).
  
  Parameters
  ----------
  matrix : numpy array of shape (number_of_docs, number_of_terms)
    The occurrence matrix. The lines stand for the documents and the columns for the terms. 
    The matrix must be binary: a one value indicates the occurrence of the term in the document, 
    and a zero indicates the opposite.
  terms : a list of strings
    The string terms. Attention: the indexes in this list do not match the matrix columns' indexes. 
    It must be provided if the similarities parameter is also provided.
  similarities : a numpy structured array of shape (number_of_terms) whose elements contain the 
  term_index and the similarity fields. The indexes of this array match the columns of the matrix parameter.
    It stores the term indexes (i.e., the index of the term in the terms parameter) and the 
    similarity value of the term. Use the the term_index field to get the term string from the 
    terms parameter.
  show : boolean
    Indicates if the plot must be shown on the screen.
  vertical_grid : boolean
    Indicates if the vertical grid lines are visible.
  save_name : string
    The name of the file to save the plot into. If None, the plot is not saved.
  '''
  _, m_axis = plt.subplots(1, figsize=fig_size)
  m_axis.imshow(matrix, aspect='auto', interpolation=None, vmin=0.0, vmax=1.0)
  m_axis.set_yticks(np.arange(0, matrix.shape[0]))
  m_axis.set_yticks(np.arange(-.5, matrix.shape[0], 0.5), minor=True)
  if vertical_grid or similarities:
    m_axis.set_xticks(np.arange(0, matrix.shape[1]))
  if vertical_grid:
    m_axis.set_xticks(np.arange(-.5, matrix.shape[1], 0.5), minor=True)
  if similarities is not None:
    labels = []
    for s in similarities:
      t_idx = s['term_index']
      sim = s['similarity']
      labels.append(f'{terms[t_idx]} ({sim:.2f})')
    m_axis.set_xticklabels(labels, rotation=90)
  m_axis.grid(which='minor', color='gray', linestyle='-', linewidth=1)
  if show:
    plt.show()
  if save_name:
    plt.savefig(save_name, bbox_inches='tight')
