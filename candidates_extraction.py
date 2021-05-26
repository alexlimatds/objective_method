import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

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

nlp = spacy.load("en_core_web_sm", disable=["ner"])
matcher = Matcher(nlp.vocab, validate=True)
p1_list = ['JJ', 'JJR', 'JJS', 'VBG', 'VBN']
p2_list = ['NN', 'NNS', 'NNP', 'NNPS', 'VBG']
pattern = []
for p1 in p1_list:
  for p2 in p2_list:
    pattern.append([{'TAG': p1, 'OP': '*'}, {'TAG': p2, 'OP': '+'}])
matcher.add('candidates', pattern)

doc = nlp(text)
matches = matcher(doc)
for match_id, start, end in matches:
  words = []
  for i in range(start, end):
    if doc[i].tag_ in ['NN', 'NNS']:
      words.append(doc[i].lemma_.lower())
    else:
      words.append(doc[i].text.lower())
  print(' '.join(words))