"""
This module comprises the code related to the transformation of string tokens into 
embedding vectors.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np

class BertVectorizer:
  """
  An instance of this class utilizes a BERT model to generate the word vectors. 
  It uses the johngiorgi/declutr-sci-base model provided by the HuggingFace library, which 
  is a BERT trained on scientific papers.
  """
  
  def __init__(self):
    model_name = 'johngiorgi/declutr-sci-base'
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)

  def get_embeddings(self, terms):
    """
    Returns the embedding vectors for a list of target terms.
    
    Parameters
    ----------
    terms:  a python list containing strings
      The targeted terms.
      
    Results
    -------
    A numpy array of shape [number of terms, embedding dimension]
      The embedding vectors, one for each term.
    """
    result = []
    for term in terms:
      inputs = self.tokenizer(term.lower(), padding=True, truncation=True, return_tensors="pt")
      # Embed the text
      with torch.no_grad():
        sequence_output = self.model(**inputs)[0]

      # Mean pool the token-level embeddings to get sentence-level embeddings
      embeddings = torch.flatten(torch.sum(
        sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
      ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9))
      result.append(embeddings.detach().numpy())

    return np.asarray(result)

# Test
def test_BertVectorizer():
  vectorizer = BertVectorizer()

  legal_terms = ['legal domain', 'legislative documents', 'eurlex']
  legal_embeddings = vectorizer.get_embeddings(legal_terms)

  ml_terms = ['deep learning', 'neural', 'cnns']
  ml_embeddings = vectorizer.get_embeddings(ml_terms)

  neutral_terms = ['task', 'house', 'desk']
  neutral_embeddings = vectorizer.get_embeddings(neutral_terms)

  def compare(term):
    print(term)
    term_embedding = vectorizer.get_embeddings([term])[0]
    for i, t in enumerate(legal_terms):
      print(f'\t<{term}> vs <{t}>: {1 - cosine(term_embedding, legal_embeddings[i])}')
    for i, t in enumerate(neutral_terms):
      print(f'\t<{term}> vs <{t}>: {1 - cosine(term_embedding, neutral_embeddings[i])}')
    for i, t in enumerate(ml_terms):
      print(f'\t<{term}> vs <{t}>: {1 - cosine(term_embedding, ml_embeddings[i])}')

  compare('legal domain')
  compare('deep learning')
  compare('staphylococcus aureus')
