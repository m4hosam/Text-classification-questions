from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Sentences that you want to compare
# Programming
# questions = ["How can I iterate over a list in Python?", 
#              "What is the syntax for looping through a Python list?"]

# Math
# questions = ["What is the principle behind Newton’s second law of motion?",
#              "Can you explain the concept of force equals mass times acceleration?"]

# Math and Geology  
questions = ["What is the significance of Euler's identity in mathematics?",
"What are the primary differences between igneous, sedimentary, and metamorphic rocks?"]


# Math and Physics 
# questions = ["Can you explain the relationship between the sides in a right-angled triangle?",
#              "Can you explain the concept of force equals mass times acceleration?"]


candidate_labels = ['Programming', 'Math', 'Physics', 'Geology']
# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute the sentence embeddings
embeddings = model.encode(questions, convert_to_tensor=True)

# Compute the cosine similarity between the sentence embeddings
cosine_scores = util.cos_sim(embeddings[0], embeddings[1])

print('Cosine similarity: {:.4f}'.format(cosine_scores.item()))


# Getting the candidate label for each question
classifier = pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli")

question_label1 = classifier(questions[0], candidate_labels)
question_label2 = classifier(questions[1], candidate_labels)

print("Question 1 candidate label: ", question_label1["labels"][0],
       ", score: ", question_label1["scores"][0])
print("Question 2 candidate label: ", question_label2["labels"][0],
       ", score: ", question_label2["scores"][0])


# Getting the keywords for each question
keywordExtractor = pipeline("token-classification",
                    model="yanekyuk/bert-keyword-extractor")

keywords1 = keywordExtractor(questions[0])
keywords2 = keywordExtractor(questions[1])

# Join subwords
keywords_joined1 = []
for keyword in keywords1:
    if keyword['word'].startswith('##'):
        keywords_joined1[-1]['word'] += keyword['word'][2:]
    else:
        keywords_joined1.append(keyword)

keywords_joined2 = []
for keyword in keywords2:
    if keyword['word'].startswith('##'):
        keywords_joined2[-1]['word'] += keyword['word'][2:]
    else:
        keywords_joined2.append(keyword)

# Print only the words
print("Question 1 keywords:")
for keyword in keywords_joined1:
    print("\t", keyword['word'])

print("Question 2 keywords:")
for keyword in keywords_joined2:
    print("\t", keyword['word'])
