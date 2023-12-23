from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# Initialize wordnet lemmatizer
wnl = WordNetLemmatizer()


stop_words = set(stopwords.words('english'))
punkt = nltk.RegexpTokenizer(r"\w+")

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


def compute_similarity_tfidf(question1, question2):
    vectorizer = TfidfVectorizer().fit_transform([question1, question2])
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    return csim[0,1]

# word embedding 
def sentence_vector(sentence, model):
    words = word_tokenize(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv.vocab]
    return np.mean(word_vectors, axis=0)

def compute_similarity(question1, question2, model):
    vector1 = sentence_vector(question1, model)
    vector2 = sentence_vector(question2, model)
    csim = cosine_similarity([vector1], [vector2])
    return csim[0,0]

question1 = "What's the weather like today?"
question2 = "How's the weather today?"
print("Question 1: ", preprocess(question2))
similarity_tdidf = compute_similarity_tfidf(question1, question2)
print("Similarity: ", similarity_tdidf)
