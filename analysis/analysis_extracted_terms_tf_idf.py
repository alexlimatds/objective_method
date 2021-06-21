"""
TF-IDF based analysis of the extracted terms.
"""
from re import I
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
    """
    An instance of this class is able to provide the TF-IDF vectors of a given corpus and the 
    respective authors' keywords. As we want to maintain these last ones as they are provided by 
    the papers, they can not be provided to the term extractor as part of the texts, so they 
    must be provided separately.

    Attributes
    ----------
    vocabulary_list_ : list of string
        The extracted vocabulary as a list.
    vocabulary_dic_ : dictionary
        The extracted vocabulary as a dictionary. The keys are the vocabulary's terms and the 
        values are the term's indexes.
    """

    def __init__(self, extractor):
        """
        Parameters
        ----------
        extractor : an object with an extract method
            The object able to extract the terms from a corpus (list of strings).
        """
        self.extractor = extractor
    
    def __count_terms__(self, corpus, known_terms):
        """
        Performs the terms' counting from a corpus and a list of authors' keywords.
        
        Parameters
        ----------
        corpus : list of string
            A set of papers.
        known_terms : list of list
            The authors' keywords. A list by each paper in corpus, following the respective 
            indexes in the corpus parameter. Provide None when a paper does not have authors' 
            keywords.
        
        Returns
        -------
        M : sparse matrix (scipy.sparse) of shape (len(corpus), n_vocabulary)
            Document-term matrix, i.e., the occurrence frequency of each term in each document. 
            The row indexing follows the indexes of the corpus parameter. The column indexing 
            follows the indexes of the vocabulary.
        """
        corpus_matrix = self.counter.transform(corpus)
        term_matrix = sps.dok_matrix(corpus_matrix.shape)
        for i,term_list in enumerate(known_terms):
            if term_list:
                for term in term_list:
                    idx = self.vocabulary_dic_[term.lower()]
                    term_matrix[i,idx] = 1
        return corpus_matrix + term_matrix.tocsr()

    def fit(self, corpus, known_terms):
        """
        Learn the vocabulary and the IDF vectors from a corpus and a list of authors' keywords.

        Parameters
        ----------
        corpus : list of string
            A set of papers.
        known_terms : list of list
            The authors' keywords. A list by each paper in corpus, following the respective 
            indexes in the corpus parameter. Provide None when a paper does not have authors' 
            keywords.
        """
        # Getting vocabulary
        self.vocabulary_list_, _ = self.extractor.extract_from_corpus(corpus, other_terms=known_terms)
        self.vocabulary_dic_ = {term: i for (i,term) in enumerate(self.vocabulary_list_)}
        # Counting terms
        self.counter = CountVectorizer(analyzer=self.extractor.extract, vocabulary=self.vocabulary_dic_)
        counting_matrix = self.__count_terms__(corpus, known_terms)
        # Learning the IDF vector
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_transformer.fit(counting_matrix)

    def transform(self, corpus, known_terms):
        """
        Generates the TF-IDF vectors for a corpus and a list of authors' keywords.

        Parameters
        ----------
        corpus : list of string
            A set of papers.
        known_terms : list of list
            The authors' keywords. A list by each paper in corpus, following the respective 
            indexes in the corpus parameter. Provide None when a paper does not have authors' 
            keywords.

        Returns
        -------
        M : sparse matrix (scipy.sparse) of shape (len(corpus), n_vocabulary)
            Document-term matrix, i.e., the TF-IDF of each term in each document. 
            The row indexing follows the indexes of the corpus parameter. The column indexing 
            follows the indexes of the vocabulary.
        """
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

plt.savefig('analysis_extracted_terms_tf_idf.pdf', bbox_inches='tight')
