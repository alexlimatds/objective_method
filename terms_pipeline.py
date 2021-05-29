"""
Module containing the pipeline code, i.e., the object in charge of generate the 
potential terms to be used in a boolean query.
"""
import candidates_extraction, terms_categorization, terms_vectorization
import time
import traceback

class TermsPipeline:
  """
  An instance of this class is able to perform the pipeline for extracting potential terms to 
  be used in a boolean query. The terms are extracted from reference corpus and with basis on 
  reference category terms. The aim is to get the terms that are more related to the provided 
  category terms.
  """
  
  def run(self, corpus, category_terms, experiment_id):
    """
    Run the pipeline. It generates a log file with the extracted terms and some results from intermediate 
    steps. The log file is name as terms_pipeline-log-<experiment_id>-<timestamp>.txt.
    
    Parameters
    ----------
    corpus : list of string
      The corpus from which the terms will be extracted. Each element in the list represents a different text/document.
    category_terms : list of string
      The category terms used as reference. Each element in the list is one caegory term.
    experiment_id : string
      Id of the experiment. It will be included in the name of the log file.
    """
    log = []
    
    try:
      log.append('Starting term extraction.')
      extractor = candidates_extraction.PosTagExtractor()
      terms = set()
      for i, text in enumerate(corpus):
        log.append(f'\tProcessing text number {i}.')
        extraction = extractor.extract(text)
        terms.update(extraction)
        log.append(f'\t\tExtracted terms ({len(extraction)} in total): {extraction}')
      terms = list(terms)
      log.append(f'\tThe whole corpus was processed. Extracted terms ({len(terms)} in total): {terms}')
      log.append('End of term extraction.')

      log.append('\nStarting term vectorization.')
      vectorizer = terms_vectorization.BertVectorizer()
      term_embeddings = vectorizer.get_embeddings(terms)
      log.append('End of term vectorization.')
      
      log.append('\nStarting term categorization.')
      log.append('\tVectorizing category terms.')
      cat_embeddings = vectorizer.get_embeddings(category_terms)
      log.append('\tClustering terms.')
      categorizer = terms_categorization.AffinityPropagationCategorizer(term_embeddings)
      log.append(f'\t\t{categorizer.number_of_clusters()} clusters created.')
      category_terms_dic = {}
      for i, e in enumerate(cat_embeddings):
        log.append(f'\tProcessing category: {category_terms[i]}.')
        indices = categorizer.term_indices_for_category(e)
        found_terms = []
        for j in indices.tolist():
          found_terms.append(terms[j])
        category_terms_dic[category_terms[i]] = found_terms
        log.append(f'\t\tFound terms ({len(found_terms)} in total): {found_terms}')
      log.append('End of term categorization.')
    
    except Exception as ex:
      log.append(f'Ocorreu uma exceção: {ex}')
      traceback.print_exc()
    
    finally:
      f = open(f"terms_pipeline-log-{experiment_id}-{time.time()}.txt", "w")
      for l in log:
        f.write(l + '\n')
      f.close()
    
# Test
def test_TermsPipeline():
  corpus = [
    '''Few-Shot Charge Prediction with Discriminative Legal Attributes. Automatic charge prediction aims to 
    predict the final charges according to the fact descriptions in criminal cases and plays a crucial role in legal 
    assistant systems. Existing works on charge prediction perform adequately on those high-frequency charges but are 
    not yet capable of predicting few-shot charges with limited cases. Moreover, these exist many confusing charge 
    pairs, whose fact descriptions are fairly similar to each other. To address these issues, we introduce several 
    discriminative attributes of charges as the internal mapping between fact descriptions and charges. These attributes 
    provide additional information for few-shot charges, as well as effective signals for distinguishing confusing charges. 
    More specifically, we propose an attribute-attentive charge prediction model to infer the attributes and charges 
    simultaneously. Experimental results on real-work datasets demonstrate that our proposed model achieves significant 
    and consistent improvements than other state-of-the-art baselines. Specifically, our model outperforms other baselines 
    by more than 50% in the few-shot scenario. Our codes and datasets can be obtained from https://github.com/thunlp/attribute_charge.''', 
    '''BillSum: A Corpus for Automatic Summarization of US Legislation. Automatic summarization methods have been 
    studied on a variety of domains, including news and scientific articles. Yet, legislation has not previously been 
    considered for this task, despite US Congress and state governments releasing tens of thousands of bills every year. 
    In this paper, we introduce BillSum, the first dataset for summarization of US Congressional and California state bills. 
    We explain the properties of the dataset that make it more challenging to process than other domains. Then, we benchmark 
    extractive methods that consider neural sentence representations and traditional contextual features. Finally, we demonstrate 
    that models built on Congressional bills can be used to summarize California billa, thus, showing that methods developed on 
    this dataset can transfer to states without human-written summaries.''']
  
  pipeline = TermsPipeline()
  pipeline.run(corpus, ['neural network', 'legal domain'], 'test')