import unittest
import numpy as np
from terms_extractors import *

class PosTagExtractorTest(unittest.TestCase):
  def test_extract(self):
    text = '''Few-Shot Charge Prediction with Discriminative Legal Attributes. Automatic charge prediction aims to 
    predict the final charges according to the fact descriptions in criminal cases and plays a crucial role in legal 
    assistant systems. Existing works on charge prediction perform adequately on those high-frequency charges but are 
    not yet capable of predicting few-shot charges with limited cases. Moreover, these exist many confusing charge 
    pairs, whose fact descriptions are fairly similar to each other. To address these issues, we introduce several 
    discriminative attributes of charges as the internal mapping between fact descriptions and charges. These attributes 
    provide additional information for few-shot charges, as well as effective signals for distinguishing confusing charges. 
    More specifically, we propose an attribute-attentive charge prediction model to infer the attributes and charges 
    simultaneously. Experimental results on real-work datasets demonstrate that our proposed model achieves significant 
    and consistent improvements than other state-of-the-art baselines. Specifically, our model outperforms other baselines 
    by more than 50% in the few-shot scenario. Our codes and datasets can be obtained from https://github.com/thunlp/attribute_charge.'''

    extractor = PosTagExtractor()
    print('PosTagExtractorTest.test_extractor log:')
    for t in extractor.extract(text):
      print(t)

  def test_get_term_doc_occurrence(self):
    terms_by_doc = [
      {'my', 'name', 'is', 'john', 'doe'},                # my name is joh doe
      {'his', 'name', 'is', 'not', 'paul', 'it', 'john'}, # his name is not paul it is john
      {'i', 'forgot', 'my', 'name'}]                      # i forgot my name
    terms = ['my', 'name', 'is', 'john', 'doe', 'his', 'not', 'paul', 'it', 'i', 'forgot']
    
    occur_matrix = get_term_doc_occurrence(terms, terms_by_doc)
    
    self.assertEqual(occur_matrix[0,0], 1) # 'my' in doc 0
    self.assertEqual(occur_matrix[1,0], 0) # 'my' in doc 1
    self.assertEqual(occur_matrix[2,0], 1) # 'my' in doc 2
    self.assertEqual(occur_matrix[0,1], 1) # 'name' in doc 0
    self.assertEqual(occur_matrix[1,1], 1) # 'name' in doc 1
    self.assertEqual(occur_matrix[2,1], 1) # 'name' in doc 2
    self.assertEqual(occur_matrix[0,2], 1) # 'is' in doc 0
    self.assertEqual(occur_matrix[1,2], 1) # 'is' in doc 1
    self.assertEqual(occur_matrix[2,2], 0) # 'is' in doc 2
  
  def test_extract_from_corpus_1(self):
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    
    extractor = PosTagExtractor()
    terms, terms_by_doc = extractor.extract_from_corpus(corpus)
    
    self.assertEqual(len(terms_by_doc), len(corpus))

  def test_extract_from_corpus_2(self):
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    known_terms = [['nick', 'nickname'], None, ['nick']]

    extractor = PosTagExtractor()
    terms, terms_by_doc = extractor.extract_from_corpus(corpus, other_terms=known_terms)
    
    self.assertEqual(len(terms_by_doc), len(corpus))
    self.assertTrue('nick' in terms)
    self.assertTrue('nickname' in terms)
    self.assertFalse(None in terms)

  def test_get_term_df(self):
    matrix = np.array([
      [1, 1, 1, 0],   # doc 0
      [1, 1, 0, 0],   # doc 1
      [1, 0, 0, 1]])   # doc 2
    
    df = get_term_df(matrix)
    
    self.assertEqual(df[0], 3)  # term 0
    self.assertEqual(df[1], 2)  # term 1
    self.assertEqual(df[2], 1)  # term 2
    self.assertEqual(df[3], 1)  # term 3
  
  def test_extract_with_df_1(self):
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    
    extractor = PosTagExtractor()
    terms, terms_df, occur_matrix = extractor.extract_with_df(corpus)
    
    self.assertEqual(occur_matrix.shape[0], len(corpus))
    self.assertEqual(len(terms_df), len(terms))
    self.assertEqual(len(terms_df), occur_matrix.shape[1])

  def test_extract_with_df_2(self):
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    known_terms = [['nick', 'nickname'], None, ['nick']]

    extractor = PosTagExtractor()
    terms, terms_df, occur_matrix = extractor.extract_with_df(corpus, other_terms=known_terms)
    
    self.assertEqual(occur_matrix.shape[0], len(corpus))
    self.assertEqual(len(terms_df), len(terms))
    self.assertEqual(len(terms_df), occur_matrix.shape[1])
    self.assertTrue('nick' in terms)
    self.assertTrue('nickname' in terms)
    self.assertFalse(None in terms)
  
  def test_terms_by_doc_1(self):
    term_idx = {
      "name": 0, 
      "nick": 1, 
      "nickname": 2
      }
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    known_terms = [['nick', 'nickname'], None, ['nick']]

    extractor = PosTagExtractor()
    terms_by_doc = extractor.terms_by_doc(term_idx, corpus, other_terms=known_terms)

    self.assertEqual(terms_by_doc[0,0], 1) # name in doc 0
    self.assertEqual(terms_by_doc[0,1], 1) # nick in doc 0
    self.assertEqual(terms_by_doc[0,2], 1) # nickname in doc 0
    self.assertEqual(terms_by_doc[1,0], 1) # name in doc 1
    self.assertEqual(terms_by_doc[1,1], 0) # nick not in doc 1
    self.assertEqual(terms_by_doc[1,2], 0) # nickname not in doc 1
    self.assertEqual(terms_by_doc[2,0], 1) # name in doc 2
    self.assertEqual(terms_by_doc[2,1], 1) # nick in doc 2
    self.assertEqual(terms_by_doc[2,2], 0) # nickname not in doc 2
  
  def test_terms_by_doc_2(self):
    term_idx = {
      "name": 0,
      "nick": 1
    }
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my nickname"]

    extractor = PosTagExtractor()
    terms_by_doc = extractor.terms_by_doc(term_idx, corpus)

    self.assertEqual(terms_by_doc[0,0], 1) # name in doc 0
    self.assertEqual(terms_by_doc[0,1], 0) # nick not in doc 0
    self.assertEqual(terms_by_doc[1,0], 1) # name in doc 1
    self.assertEqual(terms_by_doc[1,1], 0) # nick not in doc 1
    self.assertEqual(terms_by_doc[2,0], 0) # name not in doc 2
    self.assertEqual(terms_by_doc[2,1], 0) # nick not in doc 2

class TokenExtractorTest(unittest.TestCase):
  def test_extract(self):
    corpus = [
      '''Few-Shot Charge Prediction with Discriminative Legal Attributes. Automatic charge prediction aims to 
      predict the final charges according to the fact descriptions in criminal cases and plays a crucial role in legal 
      assistant systems. Experimental results on real-work datasets demonstrate that our proposed model achieves significant 
      and consistent improvements than other state-of-the-art baselines.''',
      '''Specifically, our model outperforms other baselines 
      by more than 50% in the few-shot scenario. Our codes and datasets can be obtained from 
      https://github.com/thunlp/attribute_charge. We've applied deep learning and word embeddings.''']

    extractor = TokenExtractor()
    tokens, by_doc = extractor.extract_from_corpus(corpus)
    #print(by_doc)
    self.assertEqual(len(by_doc), 2)
    print('TokenExtractorTest.test_extractor log:')
    for t in tokens:
      print(t)
  
  def test_terms_by_doc(self):
    term_idx = {
      "name": 0,
      "nickname": 1, 
      "house": 2
    }
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my nickname"]

    extractor = TokenExtractor()
    terms_by_doc = extractor.terms_by_doc(term_idx, corpus)

    self.assertEqual(terms_by_doc[0,0], 1) # name in doc 0
    self.assertEqual(terms_by_doc[0,1], 0) # nickname not in doc 0
    self.assertEqual(terms_by_doc[0,2], 0) # house not in doc 0
    self.assertEqual(terms_by_doc[1,0], 1) # name in doc 1
    self.assertEqual(terms_by_doc[1,1], 0) # nickname not in doc 1
    self.assertEqual(terms_by_doc[1,2], 0) # house not in doc 1
    self.assertEqual(terms_by_doc[2,0], 0) # name not in doc 2
    self.assertEqual(terms_by_doc[2,1], 1) # nickname in doc 2
    self.assertEqual(terms_by_doc[2,2], 0) # house not in doc 2

if __name__ == '__main__':
  unittest.main()