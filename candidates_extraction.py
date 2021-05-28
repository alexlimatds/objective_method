"""
This module comprises the code related to the extraction of candidate terms from a text.
"""
import spacy
from spacy.matcher import Matcher

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
      The text from which the canidate terms will be extracted.
      
    Returns
    -------
    A python list whose each element is a candidate term.
    """
    doc = self.nlp(text)
    matches = self.matcher(doc)
    extracted = []
    for match_id, start, end in matches:
      words = []
      for i in range(start, end):
        if doc[i].tag_ in ['NN', 'NNS']:  # if word is a noun, use its lemma (inflexed form)
          words.append(doc[i].lemma_.lower())
        else:
          words.append(doc[i].text.lower())
      extracted.append(' '.join(words))

    return extracted

# Test
def test_PosTagExtractor():
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
  for t in extractor.extract(text):
    print(t)
