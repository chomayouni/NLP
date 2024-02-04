# you will need to use pip to install the following packages: nltk, tqdm, numpy, scikit-learn
import re
import nltk
import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')

# Read the file and return a list of quotes that have the names of the characters removed
def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        quotes = [re.findall(r'"([^"]*)"', line) for line in lines]
    return [quote for sublist in quotes for quote in sublist]

# normalize the text by removing stop words, stemming, and converting to lowercase
# I chose to use the Porter Stemmer and the NLTK stop words list
def normalize_text(text_list):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    normalized_text = []
    for text in tqdm.tqdm(text_list, desc="Normalizing text"):
        words = nltk.word_tokenize(text.lower())
        words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        normalized_text.append(' '.join(words))
    return normalized_text

def get_representative_words(X, y, feature_names, n=10):
    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)

    # Get the log probabilities of each feature given each class
    log_probs = model.feature_log_prob_

    # Get the indices of the features with the highest log probability for each class
    top_feature_indices = np.argsort(-log_probs, axis=1)[:, :n]

    # Get the representative words for each class
    representative_words = feature_names[top_feature_indices]

    return representative_words

# Get the current directory
current_dir = Path(__file__).parent

# For thi code to work, the files hero_quotes.txt and villain_quotes.txt must be in the same directory as this file
# Define the file paths
hero_quotes_file = current_dir / 'hero_quotes.txt'
villain_quotes_file = current_dir / 'villain_quotes.txt'

# Read the files
hero_quotes = read_file(hero_quotes_file)
villain_quotes = read_file(villain_quotes_file)

# Print the first 5 quotes from each list
print("Sample hero quotes:\n" + '\n'.join(hero_quotes[:5]))
print("Sample villain quotes:\n" + '\n'.join(villain_quotes[:5]))

# Normalize the quotes
hero_quotes = normalize_text(hero_quotes)
villain_quotes = normalize_text(villain_quotes)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(hero_quotes + villain_quotes)

print(vectorizer.get_feature_names_out())
# print(X.toarray())

# Convert the bag of words representation to a numpy array
bag_of_words_matrix = X.toarray()
# Save the numpy array to a text file
np.savetxt("bag_of_words_matrix.txt", bag_of_words_matrix, fmt="%d")
print("The bag of words matrix has been saved to bag_of_words_matrix.txt")

# Load the bag of words matrix from the text file
bag_of_words_matrix = np.loadtxt("bag_of_words_matrix.txt", dtype=int)
print(bag_of_words_matrix)



# Create a list of categories (1 for hero quotes, 0 for villain quotes)
y = [1]*len(hero_quotes) + [0]*len(villain_quotes)

# Get the representative words for each category
representative_words = get_representative_words(X, y, vectorizer.get_feature_names_out())

print("Representative words for hero quotes:", representative_words[0])
print("Representative words for villain quotes:", representative_words[1])