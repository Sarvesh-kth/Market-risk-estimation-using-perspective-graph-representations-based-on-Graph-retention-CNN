import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

df = pd.read_csv(r'C:\Users\asus\PycharmProjects\pythonProject\FYP\DATA\Articles\A.csv', usecols=['content'])

def preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    custom_stop_words = {'agilent'}  
    stop_words = set(stopwords.words('english')).union(custom_stop_words)  
    filtered_words = [w for w in tokens if not w in stop_words and len(w) > 2]
    return filtered_words

df['processed'] = df['content'].apply(preprocess)

dictionary = corpora.Dictionary(df['processed'])
corpus = [dictionary.doc2bow(text) for text in df['processed']]

ldamodel = models.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=15)

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)