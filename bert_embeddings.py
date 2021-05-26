# TODO: description
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine

model_name = 'johngiorgi/declutr-sci-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(terms):
  '''
  Returns an python list containing the embedding vectors (torch tensors) for the target terms.
  terms  A python list with the target terms.
  '''
  result = []
  for term in terms:
    inputs = tokenizer(term.lower(), padding=True, truncation=True, return_tensors="pt")
    # Embed the text
    with torch.no_grad():
      sequence_output = model(**inputs)[0]

    # Mean pool the token-level embeddings to get sentence-level embeddings
    embeddings = torch.sum(
      sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
    ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
    result.append(embeddings)
  
  return result

# Test
legal_terms = ['legal domain', 'legislative documents', 'eurlex']
legal_embeddings = get_embeddings(legal_terms)

ml_terms = ['deep learning', 'neural', 'cnns']
ml_embeddings = get_embeddings(ml_terms)

neutral_terms = ['task', 'house', 'desk']
neutral_embeddings = get_embeddings(neutral_terms)

def compare(term):
  print(term)
  term_embedding = get_embeddings([term])[0]
  for i, t in enumerate(legal_terms):
    print(f'\t<{term}> vs <{t}>: {1 - cosine(term_embedding, legal_embeddings[i])}')
  for i, t in enumerate(neutral_terms):
    print(f'\t<{term}> vs <{t}>: {1 - cosine(term_embedding, neutral_embeddings[i])}')
  for i, t in enumerate(ml_terms):
    print(f'\t<{term}> vs <{t}>: {1 - cosine(term_embedding, ml_embeddings[i])}')

compare('legal domain')
compare('deep learning')
compare('staphylococcus aureus')
