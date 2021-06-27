"""
Functions to perform reading from the data files.
"""
import pandas as pd
import numpy as np

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
            known_terms.append([t.strip() for t in s.split(',')])
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
            known_terms.append([t.strip() for t in s.split(',')])
        else:
            known_terms.append(None)
    
    return corpus, known_terms

def merge_corpus_and_terms(texts, known_terms):
    """
    Merges a list of texts and a list of known terms. Both list must be the same 
    number of items.
    
    Parameters
    ----------
    texts : list of string
        The texts.
    known_terms : list of list of string
        The known terms for each text. Provide None when a text does not have known terms.
    """
    if len(texts) != len(known_terms):
        raise ValueError("The provided lists must have the same number of items.")
    corpus = []
    for i in range(len(texts)):
        if known_terms[i]:
            corpus.append(texts[i] + " " + ", ".join(known_terms[i]))
        else:
            corpus.append(texts[i])
    return corpus

def read_vocabulary(file_name):
    """
    Returns the vocabulary from a CSV file holding the previously extrated terms. Its 
    first line will be ignored. The remaining lines must have the following format: 
    <term>,<identification index>

    Parameters
    ----------
    file_name : string
        The path of the file to be read
    
    Returns
    -------
    A dictionary whose each item is a term (string) index (int) pair.
    """
    vocab = {}
    with open(file_name, 'r') as csvfile:
        lines = csvfile.readlines()
        skip = True # To skipe the first line
        for l in lines:
            if not skip:
                line = l.split(sep=',')
                vocab[line[0]] = int(line[1])
            else:
                skip = False
    return vocab

def read_df(file_name, vocabulary, n_docs=None):
    """
    Returns, from a CSV file, the document frequencies (DF) of terms in a vocabulary.
    Each line from the file must have the following format: <term index>,<term>,<absolute df>
    An excpetion is raised if the there is discrepancies between the term index from the 
    vocabulary and the term index from the file.

    Parameters
    ----------
    file_name : string
        The path of the file to be read
    vocabulary : a dictionary
        A dictionary whose each item is a term (string) index (int) pair.
    n_docs : float
        If provided, the document frequencies values will be divided by this parameter. In other words, 
        this function will return the relative document frequencies if this parameter is provided.
    
    Returns
    -------
    A list cotaining the read document frequencies. The list follows the 
    vocabulary's term indexes.
    """
    v_len = len(vocabulary)
    df = [None] * v_len
    with open(file_name, 'r') as csvfile:
        lines = csvfile.readlines()
        for l in lines:
            line = l.split(sep=',')
            term = line[1]
            idx = int(line[0])
            if idx != vocabulary[term]:
                raise RuntimeError(
                    f"There is a discrepancy in the index of the |{term}| term. "
                    f"Vocabulary index: {vocabulary[term]}; File index: {idx}")
            if n_docs:
                df[idx] = float(line[2]) / n_docs
            else:
                df[idx] = float(line[2])
    return df
