import openai
from openai import OpenAI
import numpy as np
import os
from scipy.special import softmax


openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def get_embeddings(texts, model="text-embedding-ada-002"):
    """
    Fetches the embedding for a given text using OpenAI's embedding model.
    """
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    embeddings = [np.array(data.embedding) for data in response.data]
    return embeddings

def cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# List of words to get embeddings for
other_words = ["queen", "princess", "monarch", "empress", "duchess", "lady", "baseball", "floor", "monitor"]
words = ["king", "man", "woman"] + other_words

# Get all embeddings in one API call
embeddings = get_embeddings(words)

# Create a mapping from word to embedding
word_to_embedding = dict(zip(words, embeddings))

# Extract embeddings
embedding_king = word_to_embedding["king"]
embedding_man = word_to_embedding["man"]
embedding_woman = word_to_embedding["woman"]

# Perform the vector arithmetic
result_vector = embedding_king - embedding_man + embedding_woman

# Calculate cosine similarities with other words
similarities = []
print("Cosine Similarities:")
for word in other_words:
    embedding_word = word_to_embedding[word]
    similarity = cosine_similarity(result_vector, embedding_word)
    similarities.append(similarity)
    print(f"Similarity with '{word}': {similarity:.4f}")

# Shift the similarities to be positive
min_similarity = min(similarities)
shifted_similarities = [sim - min_similarity for sim in similarities]

# Optionally, adjust temperature
temperature = 0.01  # Adjust this value as needed
scaled_similarities = [sim / temperature for sim in shifted_similarities]

# Apply softmax to the scaled similarities
probabilities = softmax(scaled_similarities)

# Print the softmax probabilities
print("\nSoftmax Probabilities:")
for word, probability in zip(other_words, probabilities):
    print(f"Probability for '{word}': {probability:.4f}")
