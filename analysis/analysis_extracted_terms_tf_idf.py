"""
TF-IDF based analysis of the extracted terms.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('white')

import terms_extractors
import data_readers
from analysis.tf_idf_vectorizer import TfidfVectorizer

seed_corpus, seed_known_terms = data_readers.seed_set()
population_corpus, population_known_terms = data_readers.population_set()
corpus = seed_corpus + population_corpus
known_terms = seed_known_terms + population_known_terms

extractor = terms_extractors.PosTagExtractor()
pipe = TfidfVectorizer(extractor)
pipe.fit(corpus, known_terms)

n_seed_docs = len(seed_corpus)
tf_idf = pipe.transform(seed_corpus, seed_known_terms).toarray()
v = pipe.vocabulary_list_

_, axis = plt.subplots(n_seed_docs, 3, figsize=(14, 6 * n_seed_docs))
for i in range(n_seed_docs):
    # histogram with tf-idf scores of all terms extracted from the document
    axis[i,0].set_title(f'Histogram - document {i} - {np.count_nonzero(tf_idf[i])} terms')
    min_positive = np.amin(tf_idf[i], where=(tf_idf[i] > 0), initial=1.0)
    axis[i,0].hist(tf_idf[i], range=(min_positive, tf_idf[i].max()))
    axis[i,0].grid(axis='y')
    axis[i,0].set_yscale('log')
    axis[i,0].set(xlabel='TF-IDF')

    # TF-IDF scores of the author's keywords
    axis[i,1].set_title("TF-IDF scores of authors' keywords")
    if seed_known_terms[i]: # the paper provides author's keywords
        # tf-idf scores of the authors' keywords
        keywords = []
        key_scores = []
        for k in seed_known_terms[i]:
            idx = v.index(k.lower())
            keywords.append(k)
            key_scores.append(tf_idf[i,idx])
        y_pos = range(len(keywords))
        axis[i,1].barh(y_pos, key_scores)
        axis[i,1].grid(axis='x')
        axis[i,1].set_yticks(y_pos)
        axis[i,1].set_yticklabels(keywords)
    else:
        axis[i,1].text(0.05, 0.5, "The paper doesn't provide\nauthors' keywords")

    # top terms according to tf-idf score
    n_top = 20
    top_idx = np.argsort(tf_idf[i])[-1:-(n_top+1):-1]
    terms = []
    for idx in top_idx:
        terms.append(v[idx])
    y_pos = range(n_top)
    axis[i,2].barh(y_pos, tf_idf[i,top_idx])
    axis[i,2].grid(axis='x')
    axis[i,2].set_yticks(y_pos)
    axis[i,2].set_yticklabels(terms)
    axis[i,2].set_title(f"Top {n_top} TF-IDF scores")

file_name = 'analysis_extracted_terms_tf_idf.pdf'
plt.savefig(file_name, bbox_inches='tight')
print(f'Done! Check the {file_name} file.')

