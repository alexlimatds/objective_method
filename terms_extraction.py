"""
This module comprises the code related to the extraction of candidate terms from a text.
"""
import spacy
from spacy.matcher import Matcher
import numpy as np
import re

class PosTagExtractor:
  """
  It uses POS tagging to perform extraction. The aim is to extract the noun phrases, including those 
  containing present and past participles. The desired pattern is (JJ|JJR|JJS|VBG|VBN)*(NN|NNS|NNP|NNPS|VBG)+
  """
  
  def __init__(self):
    self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
    self.matcher = Matcher(self.nlp.vocab)
    # Adding POS tag patterns to be located: noun phrases including past and present participles
    # The aim is to get the following pattern: (JJ|JJR|JJS|VBG|VBN)*(NN|NNS|NNP|NNPS|VBG)+
    p1_list = ['JJ', 'JJR', 'JJS', 'VBG', 'VBN']  # adjectives and verb participles
    p2_list = ['NN', 'NNS', 'NNP', 'NNPS', 'VBG'] # nouns and verb participles
    pattern = []
    for p1 in p1_list:
      for p2 in p2_list:
        pattern.append([{'TAG': p1, 'OP': '*'}, {'TAG': p2, 'OP': '+'}])
    self.matcher.add('candidates', pattern)

  def extract(self, text):
    """
    Performs the extraction of candidate terms from a given text. All of the candidate terms 
    are converted to lower case and the nouns are reduced to their inflexed form (lemma).
    
    Parameters
    ----------
    text : string
      The text from which the candidate terms will be extracted.
      
    Returns
    -------
    A python list whose each element is a candidate term.
    """
    invalid_regex = re.compile(r'[^a-zA-Z]+|.{1}') # tokens without letters or with just one character
    doc = self.nlp(text)
    matches = self.matcher(doc)
    extracted = []
    for match_id, start, end in matches:
      words = []
      for i in range(start, end):
        if not invalid_regex.fullmatch(doc[i].text):
          if doc[i].tag_ in ['NN', 'NNS']:  # if word is a noun, use its lemma (inflexed form)
            words.append(doc[i].lemma_.lower())
          else:
            words.append(doc[i].text.lower())
      if len(words) > 0:
        extracted.append(' '.join(words))

    return extracted

  def extract_from_corpus(self, corpus, other_terms=None):
    """
    Performs the extraction of candidate terms from a given text. All of the candidate terms 
    are converted to lower case and the nouns are reduced to their inflexed form (lemma). 

    Parameters
    ----------
    corpus : list of string
      A list whose each element is a document in the corpus.
    other_terms : list of list of strings
      The previously known terms for each document. Provide a list for each document. For 
      documents without known terms, provide None as value.

    Returns
    -------
    terms : list of string
      A list whose each element is an extracted term.
    terms_by_doc : list of set of strings
      The term set of each document. The list follows the same document indexes from 
      the corpus parameter.
    """
    terms = set()
    terms_by_doc = [None] * len(corpus)
    for i, text in enumerate(corpus):
      extraction = self.extract(text)
      if other_terms and other_terms[i]:
        extraction.extend([t.lower() for t in other_terms[i]])
      terms_by_doc[i] = set(extraction)
      terms.update(extraction)
    terms = list(terms)

    return terms, terms_by_doc
  
  def terms_by_doc(self, term_idx, corpus, other_terms=None):
    """
    Generate the occurence matrix of a specific term set in a corpus.
    
    Parameters
    ----------
    term_idx : dictionary with string for key and int for value
      A dictionary storing the reference indexes of the term set. Each key (string) is a term 
      and each value is the reference index of such key.
    corpus : a list of strings.
      A list whose each element is a document.
    
    Returns
    -------
    A numpy matrix with shape (len(terms_by_doc), len(terms)). The 1 value indicates a occurrence of the 
    term in the document. The zero value indicates the opposite case. The column indexes follow the 
    indexes from the term_idx parameter. The row indexes follow the index from the corpus parameter.
    """
    terms_by_doc = np.zeros((len(corpus), len(term_idx)))
    for i, text in enumerate(corpus):
      extraction = self.extract(text)
      if other_terms and other_terms[i]:
        extraction.extend([t.lower() for t in other_terms[i]])
      for e in extraction:
        idx = term_idx.get(e)
        if idx is not None:
          terms_by_doc[i, idx] = 1

    return terms_by_doc

  def extract_with_df(self, corpus, other_terms=None):
    """
    Performs the extraction of candidate terms from a given text. All of the candidate terms 
    are converted to lower case and the nouns are reduced to their inflexed form (lemma). 
    It also generates the occurrence matrix of the terms in the documents and the term document 
    frequencies.

    Parameters
    ----------
    corpus : list of string
      A list whose each element is a document in the corpus.
    other_terms : list of list of string
      The previously known terms for each document. Provide a list for each document. For 
      documents without known terms, provide None as value.

    Returns
    -------
    terms : list of string
      A list whose each element is an extracted term.
    term_df : numpy array of shape (len(terms))
      The document frequency (DF) of each term. Each index represents the DF of the respective 
      term in the returned terms list.
    occurrence_matrix : numpy array of shape (number of documents, number of terms)
      The same matrix returned by the __term_doc_occurrence__ function.
    """
    terms, terms_by_doc = self.extract_from_corpus(corpus, other_terms=other_terms)
    
    occurrence_matrix = get_term_doc_occurrence(terms, terms_by_doc)
    term_df = get_term_df(occurrence_matrix)
    
    return terms, term_df, occurrence_matrix

def get_term_df(occurrence_matrix):
  """
  Calculates the document frequency of each term.
  
  Parameters
  ----------
  occurrence_matrix : A numpy matrix with shape (number of documents, number of terms)
    The occurrence matrix generated by the __term_doc_occurrence__ function.
  
  Returns
  -------
  numpy array of shape (number of terms) containing  the document frequency of each term.
  """
  return np.sum(occurrence_matrix, axis=0)
    
def get_term_doc_occurrence(terms, terms_by_doc):
  """
  Generate the occurence matrix of terms in a corpus.
  
  Parameters
  ----------
  terms : list of string
    A list containing the terms of the corpus. Each term must occur once and the indexes of the 
    list will be used ans the column indexes the result matrix.
  terms_by_doc : a list of sets of strings.
    Each list element represents a document in the corpus, while each set contains the terms in the 
    respective document. The list indexes will be used as the row indexes of the result matrix.
  
  Returns
  -------
  A numpy matrix with shape (len(terms_by_doc), len(terms)). The 1 value indicates a occurrence of the 
  term in the document. The zero value indicates the opposite case.
  """
  doc_term_occurrence = np.zeros((len(terms_by_doc), len(terms)), dtype=np.int32)
  for i, doc in enumerate(terms_by_doc):
    for j, term in enumerate(terms):
      if term in terms_by_doc[i]:
        doc_term_occurrence[i, j] = 1
  
  return doc_term_occurrence
  