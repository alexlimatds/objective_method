# Text preprocessing functions
import re
import pandas as pd

'''
Performs text preprocessing
  text  - A pandas column containing text.
'''
def preprocess(text):
  # converting to lower case
  text = text.apply(lambda x: x.lower())
  # removing symbols and punctuation
  text = text.apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.#:]', '', x))
  # removing numbers
  text = text.apply(lambda x: re.sub(r'\d+', '', x))
  # removing additional spaces
  text = text.apply(lambda x: re.sub(r' {2,}', ' ', x))
  text = text.apply(lambda x: x.strip().replace('\n', ' '))
  
  return text
