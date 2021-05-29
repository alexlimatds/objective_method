import terms_pipeline
import pandas as pd

df = pd.read_csv('seed_set.csv')
corpus = df['title'].str.cat(df[['abstract', 'keywords']], sep=' ', na_rep='').tolist()

pipeline = terms_pipeline.TermsPipeline()
pipeline.run(corpus, ['deep learning', 'legal domain'], 'experiment_01')