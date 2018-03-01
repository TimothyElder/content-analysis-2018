pip install -U git+git://github.com/Computational-Content-Analysis-2018/lucem_illud.git
import lucem_illud
import nltk

plos_df = pd.read_pickle('../data/plos_sample.pk1')

plosTokens = nltk.word_tokenize(plos_df['Article Contents'].sum())

plos_df['tokenized_text'] = plos_df['Article Contents'].apply(lambda x: nltk.word_tokenize(x))

plos_df['word_counts'] = plos_df['tokenized_text'].apply(lambda x: len(x))

plos_df.to_pickle('../data/plos_sample.pk2')



