"""
Comparison between the authors' keywords and the other terms in the same document.
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
rcParams.update({'figure.autolayout': True})
sns.set_style('white')
import terms_extraction
import data_readers
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

seed_corpus, seed_known_terms = data_readers.seed_set()
population_corpus, population_known_terms = data_readers.population_set()
corpus = seed_corpus + population_corpus
known_terms = seed_known_terms + population_known_terms

extractor = terms_extraction.PosTagExtractor()
v, _ = extractor.extract_from_corpus(corpus, other_terms=known_terms)
pipe = Pipeline([
    ('count', CountVectorizer(analyzer=extractor.extract, vocabulary=v)), 
    ('tfidf', TfidfTransformer())
])
pipe.fit(corpus)

n_seed_docs = len(seed_corpus)
n_seed_keywords_lists = len(seed_known_terms) - seed_known_terms.count(None)
tf_idf = pipe.transform(seed_corpus).toarray()

_, axis = plt.subplots(n_seed_keywords_lists, 2, figsize=(10, 4 * n_seed_keywords_lists))
plot_count = 0
for i in range(n_seed_docs):
    if seed_known_terms[i]:
        # histogram with tf-idf scores of all terms in the document
        axis[plot_count,0].set_title(f'TF-IDF histogram - document {i+1}')
        axis[plot_count,0].hist(tf_idf[i])
        axis[plot_count,0].grid(axis='y')
        axis[plot_count,0].set_yscale('log')
        axis[plot_count,0].set(xlabel='Term TF-IDF', ylabel='Frequency')

        # tf-idf scores of the authors' keywords
        keywords = []
        key_scores = []
        for k in seed_known_terms[i]:
            idx = v.index(k.lower())
            keywords.append(k)
            key_scores.append(tf_idf[i,idx])
        y_pos = range(len(keywords))
        axis[plot_count,1].barh(y_pos, key_scores)
        axis[plot_count,1].set_yticks(y_pos)
        axis[plot_count,1].set_yticklabels(keywords)
        axis[plot_count,1].set_title("TF-IDF scores of authors' keywords")

        plot_count += 1

plt.savefig('analysis_author_keywrods_tf-idf.pdf', bbox_inches='tight')
