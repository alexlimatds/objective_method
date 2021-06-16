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
# TODO modify the current pipeline to take the author keywords in account
pipe = Pipeline([
    ('count', CountVectorizer(analyzer=extractor.extract, vocabulary=v)), 
    ('tfidf', TfidfTransformer())
])
pipe.fit(corpus)

n_seed_docs = len(seed_corpus)
n_seed_keywords_lists = len(seed_known_terms) - seed_known_terms.count(None)
tf_idf = pipe.transform(seed_corpus).toarray()

_, axis = plt.subplots(n_seed_keywords_lists, 3, figsize=(14, 4 * n_seed_keywords_lists))
plot_count = 0
for i in range(n_seed_docs):
    if seed_known_terms[i]: # the paper provides author's keywords
        # histogram with tf-idf scores of all terms in the document
        axis[plot_count,0].set_title(f'TF-IDF histogram - document {i}')
        axis[plot_count,0].hist(tf_idf[i])
        axis[plot_count,0].grid(axis='y')
        axis[plot_count,0].set_yscale('log')
        axis[plot_count,0].set(xlabel='TF-IDF', ylabel='Frequency')

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

        # top terms according to tf-idf score
        n_top = 10
        top_idx = np.argsort(tf_idf[i])[-1:-(n_top+1):-1]
        terms = []
        for idx in top_idx:
            terms.append(v[idx])
        y_pos = range(n_top)
        axis[plot_count,2].barh(y_pos, tf_idf[i,top_idx])
        axis[plot_count,2].set_yticks(y_pos)
        axis[plot_count,2].set_yticklabels(terms)
        axis[plot_count,2].set_title(f"Top {n_top} TF-IDF scores")

        plot_count += 1

plt.savefig('analysis_author_keywrods_tf-idf.pdf', bbox_inches='tight')
