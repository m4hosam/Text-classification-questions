import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TurkishStemmer import TurkishStemmer
from gensim.models import KeyedVectors


# Download necessary resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = TurkishStemmer()
stop_words = set(stopwords.words('turkish'))
punkt = nltk.RegexpTokenizer(r"\w+")
# Word2Vec model
word_vectors = KeyedVectors.load_word2vec_format('word2vec-trmodel', binary=True)

def preprocess(text):
    # Tokenization using Regex Tokenizer
    tokens = punkt.tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Stopword removal
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def calculate_similarity(question1, question2):
    # Preprocess the questions
    question1 = preprocess(question1)
    question2 = preprocess(question2)

    # Vectorize the questions
    vectorizer = TfidfVectorizer().fit_transform([' '.join(question1), ' '.join(question2)])
    
    # Calculate and return the cosine similarity
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return similarity

def calculate_similarity_model(question1, question2):
    question1 = preprocess(question1)
    question2 = preprocess(question2)

    # Calculate the similarity between the questions using word embeddings
    similarity = word_vectors.n_similarity(question1, question2)
    return similarity

# Test the function
question3 = "Python dilinde nasıl bir döngü oluşturabilirim?"
question4 = "java'da bir dizi nasıl yazılır?"


similarity_score = calculate_similarity(question3, question4)
print(f"Similarity between question3 and question4: {similarity_score[0][0]}")

similarity_score_2 = calculate_similarity_model(question3, question4)
print(f"Similarity with model between question3 and question4: {similarity_score_2}")
