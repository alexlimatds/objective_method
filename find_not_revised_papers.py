import pandas as pd
import re

def normalize_string(str):
  str = str.lower().strip().replace('\n', ' ')
  str = re.sub(r' {2,}', ' ', str)
  str = re.sub(r'[^0-9a-z #+_/(){}\[\]\|@,;:]', '', str)
  return str

'''
Identifies the papers obtained through a search at IEEE Xplore but that aren't revised yet.
The revision aims to indicate if a paper is related or not to the topic of interest.
The revised papers are in the revised_papers.csv file.
  df_search   The Pandas DataFrame containing the results from the search at IEEE Xplore
'''
def find(df_search):
  df_revised = pd.read_csv('revised_papers.csv')
  df_revised['Title'] = df_revised['Title'].apply(lambda t: normalize_string(t))

  not_found = []
  warnings = []
  for index, row in df_search.iterrows():
    # Searching by DOI
    result = df_revised[df_revised['Identifier'] == row['DOI']]
    if result.shape[0] == 0: # DOI not found
      # Searching by title
      result = df_revised[df_revised['Title'] == normalize_string(row['Document Title'])]
      if result.shape[0] == 0: # Title not found
        not_found.append(f'<{row["DOI"]}>  <{row["Document Title"]}>')
    elif result.shape[0] != 1:
      warnings.append(f'WARNING: found more than one occurrence for the paper <{row["DOI"]}>  <{row["Document Title"]}>')

  f = open("find_not_revised_papers.out.txt", "w")
  for l in warnings:
    f.write(l + '\n')
  for l in not_found:
    f.write(l + '\n')
  f.close()

manual_search = pd.read_csv('export2021.05.10-11.50.05.csv') # The papers obtained through a manuel search at IEEE Xplore
find(manual_search)