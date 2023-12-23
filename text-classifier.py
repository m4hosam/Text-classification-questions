from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli")

keywordExtractor = pipeline("token-classification",
                    model="yanekyuk/bert-keyword-extractor")

sequence_to_classify = "How does the concept of recursion in computer science relate to the fractal geometry found in nature? "
candidate_labels = ['Programming', 'Math', 'Psychology', 'Geology']
k = classifier(sequence_to_classify, candidate_labels)
keywords = keywordExtractor(sequence_to_classify)
# print(k)
print(k["labels"][0], "score: ", k["scores"][0])
# Join subwords
keywords_joined = []
for keyword in keywords:
    if keyword['word'].startswith('##'):
        keywords_joined[-1]['word'] += keyword['word'][2:]
    else:
        keywords_joined.append(keyword)

# Print only the words
for keyword in keywords_joined:
    print(keyword['word'])
# print(keywords)