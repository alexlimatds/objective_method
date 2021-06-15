"""
Functions to perform reading from the data files.
"""
import pandas as pd

def seed_set():
    """
    Reads the data from the seed dataset.

    Returns
    -------
    corpus : list of strings
        The seed corpus. Each element in the list is a document.
    known_terms : list
        The keywords or keyterms provided by the documents. Each element in the list 
        may be a list of strings, when the document provides a keyword set, or None.
    """
    data = pd.read_csv('seed_set.csv')
    corpus = data['title'].str.cat(data['abstract'], sep='. ', na_rep='').tolist()
    # not every paper provide a keyword list
    known_terms = []
    for s in data['keywords'].to_list():
        if type(s) == str:
            known_terms.append(s.split(','))
        else:
            known_terms.append(None)
    
    return corpus, known_terms

def population_set():
    """
    Reads the data from the population dataset.

    Returns
    -------
    corpus : list of strings
        The population corpus. Each element in the list is a document.
    known_terms : list
        The keywords or keyterms provided by the documents. Each element in the list 
        may be a list of strings, when the document provides a keyword set, or None.
    """
    data = pd.read_csv('population_set_ieee.csv')
    corpus = data['title'].str.cat(data['abstract'], sep='. ', na_rep='').tolist()
    # not every paper provide a keyword list
    known_terms = []
    for s in data['terms'].to_list():
        if type(s) == str:
            known_terms.append(s.split(','))
        else:
            known_terms.append(None)
    
    return corpus, known_terms
