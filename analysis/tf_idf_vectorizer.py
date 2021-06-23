from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
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