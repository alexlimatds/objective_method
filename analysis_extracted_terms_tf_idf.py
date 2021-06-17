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
import scipy.sparse as sps

class TfidfVectorizer:

    def __init__(self, extractor):
        self.extractor = extractor
    
    def __count_terms__(self, corpus, known_terms):
        corpus_matrix = self.counter.transform(corpus)
        term_matrix = sps.dok_matrix(corpus_matrix.shape)
        for i,term_list in enumerate(known_terms):
            if term_list:
                for term in term_list:
                    idx = self.vocabulary_dic_[term.lower()]
                    term_matrix[i,idx] = 1
        return corpus_matrix + term_matrix.tocsr()

    def fit(self, corpus, known_terms):
        # Getting vocabulary
        extractor = terms_extraction.PosTagExtractor()
        self.vocabulary_list_, _ = extractor.extract_from_corpus(corpus, other_terms=known_terms)
        self.vocabulary_dic_ = {term: i for (i,term) in enumerate(self.vocabulary_list_)}
        # Counting terms
        self.counter = CountVectorizer(analyzer=extractor.extract, vocabulary=self.vocabulary_dic_)
        counting_matrix = self.__count_terms__(corpus, known_terms)
        # Learning the IDF vector
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_transformer.fit(counting_matrix)

    def transform(self, corpus, known_terms):
        counting_matrix = self.__count_terms__(corpus, known_terms)
        return self.tfidf_transformer.transform(counting_matrix)


seed_corpus, seed_known_terms = data_readers.seed_set()
population_corpus, population_known_terms = data_readers.population_set()
corpus = seed_corpus + population_corpus
known_terms = seed_known_terms + population_known_terms

extractor = terms_extraction.PosTagExtractor()
pipe = TfidfVectorizer(extractor)
pipe.fit(corpus, known_terms)

n_seed_docs = len(seed_corpus)
n_seed_keywords_lists = len(seed_known_terms) - seed_known_terms.count(None)
#tf_idf = pipe.transform(seed_corpus).toarray()
tf_idf = pipe.transform(seed_corpus, seed_known_terms).toarray()
v = pipe.vocabulary_list_

_, axis = plt.subplots(n_seed_keywords_lists, 3, figsize=(14, 6 * n_seed_keywords_lists))
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
        n_top = 20
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
