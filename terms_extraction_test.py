import unittest
import numpy as np
import terms_extraction

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

    extractor = terms_extraction.PosTagExtractor()
    print('test_extractor log:')
    for t in extractor.extract(text):
      print(t)

  def test__term_doc_occurrence__(self):
    terms_by_doc = [
      {'my', 'name', 'is', 'john', 'doe'},                # my name is joh doe
      {'his', 'name', 'is', 'not', 'paul', 'it', 'john'}, # his name is not paul it is john
      {'i', 'forgot', 'my', 'name'}]                      # i forgot my name
    terms = ['my', 'name', 'is', 'john', 'doe', 'his', 'not', 'paul', 'it', 'i', 'forgot']
    
    occur_matrix = terms_extraction.__term_doc_occurrence__(terms, terms_by_doc)
    
    self.assertEqual(occur_matrix[0,0], 1) # 'my' in doc 0
    self.assertEqual(occur_matrix[1,0], 0) # 'my' in doc 1
    self.assertEqual(occur_matrix[2,0], 1) # 'my' in doc 2
    self.assertEqual(occur_matrix[0,1], 1) # 'name' in doc 0
    self.assertEqual(occur_matrix[1,1], 1) # 'name' in doc 1
    self.assertEqual(occur_matrix[2,1], 1) # 'name' in doc 2
    self.assertEqual(occur_matrix[0,2], 1) # 'is' in doc 0
    self.assertEqual(occur_matrix[1,2], 1) # 'is' in doc 1
    self.assertEqual(occur_matrix[2,2], 0) # 'is' in doc 2
    
  def test__term_df__(self):
    matrix = np.array([
      [1, 1, 1, 0],   # doc 0
      [1, 1, 0, 0],   # doc 1
      [1, 0, 0, 1]])   # doc 2
    
    df = terms_extraction.__term_df__(matrix)
    
    self.assertEqual(df[0], 3)  # term 0
    self.assertEqual(df[1], 2)  # term 1
    self.assertEqual(df[2], 1)  # term 2
    self.assertEqual(df[3], 1)  # term 3
  
  def test_extract_with_df_1(self):
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    
    extractor = terms_extraction.PosTagExtractor()
    terms, terms_df, occur_matrix = extractor.extract_with_df(corpus)
    
    self.assertEqual(occur_matrix.shape[0], len(corpus))
    self.assertEqual(len(terms_df), len(terms))
    self.assertEqual(len(terms_df), occur_matrix.shape[1])

  def test_extract_with_df_2(self):
    corpus = [
      "My name is John Doe", 
      "His name is not Paul, it is John", 
      "I forgot my name"]
    known_terms = ['name', 'nickname', 'nick']

    extractor = terms_extraction.PosTagExtractor()
    terms, terms_df, occur_matrix = extractor.extract_with_df(corpus, other_terms=known_terms)
    
    self.assertEqual(occur_matrix.shape[0], len(corpus))
    self.assertEqual(len(terms_df), len(terms))
    self.assertEqual(len(terms_df), occur_matrix.shape[1])
    for t in known_terms:
      self.assertTrue(t in terms)
  
if __name__ == '__main__':
  unittest.main()