from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.test.utils import common_texts
import gensim.downloader as api


wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punkt = nltk.RegexpTokenizer(r"\w+")

original_text = """In computer architecture, multithreading is
                the ability of a central processing unit (CPU)
                (or a single core in a multi-core processor) to 
                provide multiple threads of execution concurrently,
                supported by the operating system. This approach
                differs from multiprocessing. In a multithreaded
                application, the threads share the resources of
                a single or multiple cores, which include the
                computing units, the CPU caches, and the 
                translation lookaside buffer (TLB)."""

candidate_labels = ['Programming', 'Math', 'Psychology', 'Geology']


def preprocess(text):
    # Tokenization using Regex Tokenizer
    tokens = punkt.tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Stopword removal
    tokens = [token for token in tokens if token not in stop_words]

    # lemmatization
    tokens = [wnl.lemmatize(token) for token in tokens]
    
    return tokens

tokens = preprocess(original_text)
model = Word2Vec.load("word2vec-google-news-300.gz")


def transform_to_vector(text):
    vector = np.zeros(100)  # assuming vector_size=100
    words = text.split()
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
    return vector / len(words)

# Transform your original text to vector
original_text_vector = transform_to_vector(original_text)

# Transform your labels to vectors
labels_vector = [transform_to_vector(label) for label in candidate_labels]

# Train a classifier
clf = RandomForestClassifier()
clf.fit(labels_vector, candidate_labels)

# Predict the label of the original text
predicted_label = clf.predict([original_text_vector])

print(predicted_label)
