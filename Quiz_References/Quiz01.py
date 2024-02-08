from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from fractions import Fraction
from numpy.linalg import svd
import numpy as np


def calculate_unigram_perplexity(sequence, unigram_counts, total_count):
    # Convert the sequence to lower case and split it into words
    words = sequence.lower().split()

    # Calculate the unigram probabilities
    probabilities = [unigram_counts[word] / total_count for word in words if word in unigram_counts]
    # If a word in the sequence is not in the unigram_counts, we skip it

    # Calculate the perplexity
    perplexity = np.exp(-np.sum(np.log(probabilities)) / len(words))

    return perplexity

# These are just example counts, replace them with your actual counts
unigram_counts = {"the": 1000, "a": 500, "an": 300}
total_count = sum(unigram_counts.values())

sequence = "The a an"

perplexity = calculate_unigram_perplexity(sequence, unigram_counts, total_count)

# print("Unigram Perplexity:", perplexity)


def calculate_bigram_perplexity(sequence, bigram_counts, unigram_counts):
    # Convert the sequence to lower case and split it into words
    words = sequence.lower().split()

    # Calculate the bigram probabilities
    probabilities = [bigram_counts[(words[i-1], words[i])] / unigram_counts[words[i-1]] for i in range(1, len(words)) if (words[i-1], words[i]) in bigram_counts]

    # If a bigram in the sequence is not in the bigram_counts, we skip it

    # Calculate the perplexity
    perplexity = np.exp(-np.sum(np.log(probabilities)) / (len(words) - 1))

    return perplexity

# These are just example counts, replace them with your actual counts
unigram_counts = {"<s>": 100, "the": 75, "cat": 20}
bigram_counts = {("<s>", "the"): 25, ("the", "cat"): 15, ("cat", "sat"): 10}

sequence = "<s> the cat sat"

perplexity = calculate_bigram_perplexity(sequence, bigram_counts, unigram_counts)

# print("Bigram Perplexity:", perplexity)

def calculate_distances(doc1, doc2):
    # Convert the documents into a "bag of words" format
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([doc1, doc2]).toarray()

    # Print the vectors
    print("Vector for doc1:", X[0])
    print("Vector for doc2:", X[1])

    # Calculate the Jaccard similarity
    jaccard_similarity = jaccard_score(X[0], X[1], average='micro')

    # Calculate the Euclidean distance
    euclidean_distance = euclidean_distances(X[0].reshape(1, -1), X[1].reshape(1, -1))[0][0]

    # Calculate the Cosine distance
    cosine_distance = cosine_distances(X[0].reshape(1, -1), X[1].reshape(1, -1))[0][0]

    print("Jaccard Similarity:", jaccard_similarity)
    print("Euclidean Distance:", euclidean_distance)
    print("Cosine Distance:", cosine_distance)
    print("\n")

    # Convert the results to fractions and round to the nearest two decimal places
    jaccard_similarity = round(Fraction(jaccard_similarity).limit_denominator(), 2)
    euclidean_distance = round(Fraction(euclidean_distance).limit_denominator(), 2)
    cosine_distance = round(Fraction(cosine_distance).limit_denominator(), 2)

    return jaccard_similarity, euclidean_distance, cosine_distance

# doc1 = "the dog chased the ball"
# doc2 = "the person chased the dog"
doc1 = "the computer and the keyboard and the mouse"
doc2 = "the freedom and the hope and the joy"
jaccard_similarity, euclidean_distance, cosine_distance = calculate_distances(doc1, doc2)

print("Jaccard Similarity:", jaccard_similarity)
print("Euclidean Distance:", euclidean_distance)
print("Cosine Distance:", cosine_distance)

##TF-IDF --- couldn't get the code to work

# def calculate_tfidf(docs):
#     # Initialize the TfidfVectorizer
#     vectorizer = TfidfVectorizer()

#     # Compute the TF-IDF
#     tfidf_matrix = vectorizer.fit_transform(docs).toarray()

#     # Get the feature names (words)
#     feature_names = vectorizer.get_feature_names_out()

#     # Get the IDF scores
#     idf_scores = vectorizer.idf_

#     return tfidf_matrix, feature_names, idf_scores

# docs = ["The path ran through the woods.", "the runner ran down the road", "the woods ran along the side of the road"]

# tfidf_matrix, feature_names, idf_scores = calculate_tfidf(docs)

# # Print the TF-IDF vectors
# for i, doc in enumerate(docs):
#     print(f"TF-IDF vector for doc{i+1}:")
#     print(tfidf_matrix[i])

# # Print the IDF scores
# print("\nIDF scores:")
# for word, score in zip(feature_names, idf_scores):
#     print(f"{word}: {score}")


#LSA/LSI--Latent Semantic Analysis/Indexing
# Define a matrix
# Don't forget to verify that you have the correct size matrix
A = np.array([[2,1,0,0,0], [1,1,1,0,0], [0,0,1,1,1], [0,0,0,1,2]])

print("Matrix A:\n", A)

# Calculate the singular value decomposition
U, S, VT = np.linalg.svd(A)
# U.round(2), S.round(2), VT.round(2)

print("U:\n", U.round(2) )
print("S:\n", S.round(2))
print("VT:\n", VT.round(2))

# Define the rank for the reduced approximation
k = 2

# Keep only the first k singular values/vectors
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

print("U_k:\n", U_k.round(2))
print("S_k:\n", S_k.round(2))
print("VT_k:\n", VT_k.round(2))

# Calculate the reduced-rank approximation of A
A_k = U_k @ S_k @ VT_k
print("\n ----------------------------" )
print("Reduced-rank approximation of A using a K value of " + str(k) + ":\n", A_k.round(2))
print("\nAccuracy goes down as the depending on the size of K" )

