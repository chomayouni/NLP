# you will need to use pip to install the following packages: nltk, tqdm, numpy, scikit-learn
import re
import nltk
import tqdm
import argparse
import string
import numpy as np
import sys
import pandas as pd

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
current_dir = Path(__file__).parent

# Read the file and return a list of quotes that have the names of the characters removed
def read_file(file_name, standard, custom):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if standard:
            quotes = [re.findall(r'.*', line) for line in lines]
            return [quote for sublist in quotes for quote in sublist]
        elif custom:
            quotes = [re.findall(r'"([^"]*)"', line) for line in lines]
            return [quote for sublist in quotes for quote in sublist]
        else:
            print("Please specify a normalization method")
            return None

def custom_normalize_text(text_list):
    lemmatizer = WordNetLemmatizer()
    stop_words = ["0o", "you" "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
    normalized_text = []
    for text in tqdm.tqdm(text_list, desc="Normalizing text"):
        words = nltk.word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        words = [''.join(c for c in word if c not in string.punctuation) for word in words]
        words = [''.join(c for c in word if not c.isdigit()) for word in words]
        words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        normalized_text.append(' '.join(words))
    return normalized_text

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

def setup_naive_bayes(hero_quotes, villain_quotes, data_type='count'):

    # Combine the quotes from both categories
    all_quotes = hero_quotes + villain_quotes

    # Create a bag of words representation of the quotes
    if data_type == 'count':
        vectorizer = CountVectorizer()
    elif data_type == 'binary':
        vectorizer = CountVectorizer(binary=True)
    elif data_type == 'tfidf':
        vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the combined quotes and transform the quotes into a bag of words matrix
    X = vectorizer.fit_transform(all_quotes)
    vocabulary = vectorizer.get_feature_names_out()
    bag_of_words_matrix = X.toarray()

    # Split the bag of words matrix back into hero and villain quotes
    hero_bow = bag_of_words_matrix[:len(hero_quotes)]
    villain_bow = bag_of_words_matrix[len(villain_quotes):]

    # Calculate the total number of words in each category
    total_hero_words = np.sum(hero_bow)
    total_villain_words = np.sum(villain_bow)
    total_words = len(vocabulary)

    # Calculate the number of documents and the average number of tokens per document for each category
    hero_docs = len(hero_quotes)
    villain_docs = len(villain_quotes)
    avg_hero_tokens = total_hero_words / hero_docs
    avg_villain_tokens = total_villain_words / villain_docs

    return hero_bow, villain_bow, total_hero_words, total_villain_words, total_words, vocabulary, vectorizer, bag_of_words_matrix, hero_docs, villain_docs, avg_hero_tokens, avg_villain_tokens

def calculate_naive_bayes(hero_bow, villain_bow, total_hero_words, total_villain_words, total_words, vocabulary, vectorizer):

    # Initialize dictionaries to hold the word probabilities for each category
    hero_word_prob = defaultdict(float)
    villain_word_prob = defaultdict(float)

    # Calculate the log probability of each word in each category
    for word in vocabulary:
        # Get the index of the word in the vocabulary
        word_index = vectorizer.vocabulary_.get(word, 0)

        # Calculate the word probabilities for the hero category
        # This is the number of times the word appears in the hero quotes (plus 1 for smoothing),
        # divided by the total number of words in the hero quotes (plus the total number of unique words for smoothing)
        # The np.log function is applied to convert the probability to a log probability
        hero_word_count = np.sum(hero_bow[:, word_index])
        hero_word_prob[word] = np.log((hero_word_count + 1) / (total_hero_words + total_words))

        # Calculate the word probabilities for the villain category
         # This is the number of times the word appears in the villain quotes (plus 1 for smoothing),
        # divided by the total number of words in the villain quotes (plus the total number of unique words for smoothing)
        # The np.log function is applied to convert the probability to a log probability
        villain_word_count = np.sum(villain_bow[:, word_index])
        villain_word_prob[word] = np.log((villain_word_count + 1) / (total_villain_words + total_words))

    return hero_word_prob, villain_word_prob


def main():
    # Create a parser
    parser = argparse.ArgumentParser(description="Modify text normalization process")
    parser.add_argument("-c", "--custom", action='store_true', help="Uses custom normalization on the text")
    parser.add_argument("-s", "--standard", action='store_true', help="Uses standard normalization on the text")
    parser.add_argument('--data_type', type=str, default='count', choices=['count', 'binary', 'tfidf'], help='The type of data to use when computing probabilities.')
    args = parser.parse_args()

    if not args.standard and not args.custom:
        print("Please select either --standard or --custom.")
        sys.exit()
    else:
        #----------------ReadFile----------------
        # For thi code to work, the files hero_quotes.txt and villain_quotes.txt must be in the same directory as this file
        # Define the file paths
        hero_quotes_file = current_dir / 'hero_quotes.txt'
        villain_quotes_file = current_dir / 'villain_quotes.txt'
        hero_quotes = read_file(hero_quotes_file, standard=args.standard, custom=args.custom)
        villain_quotes = read_file(villain_quotes_file, standard=args.standard, custom=args.custom)

        #----------------NormalizeText----------------
        # Normalize the quotes
        if args.custom:
            hero_quotes = custom_normalize_text(hero_quotes)
            villain_quotes = custom_normalize_text(villain_quotes)
        elif args.standard:
            hero_quotes = normalize_text(hero_quotes)
            villain_quotes = normalize_text(villain_quotes)

        # Print the first 5 quotes from each list
        print("Sample hero quotes:\n" + '\n'.join(hero_quotes[:5]))
        print("Sample villain quotes:\n" + '\n'.join(villain_quotes[:5]))


        #----------------NaiveBayes----------------
        # Train the Naive Bayes model
        hero_bow, villain_bow, total_hero_words, total_villain_words, total_words, vocabulary, vectorizer, bag_of_words_matrix, hero_docs, villain_docs, avg_hero_tokens, avg_villain_tokens = setup_naive_bayes(hero_quotes, villain_quotes)
        hero_word_prob, villain_word_prob = calculate_naive_bayes(hero_bow, villain_bow, total_hero_words, total_villain_words, total_words, vocabulary, vectorizer)

        # Print the stats for the dataset
        # These are the stats that will be used to calculate the log likelihood ratios
        print(f"Number of quotes: {len(hero_quotes) + len(villain_quotes)}")
        print(f"Number of tokens: {total_hero_words + total_villain_words} (Number of Hero tokens: {total_hero_words}, Number of Villain tokens: {total_villain_words})")
        print(f"Number of types: {total_words}")
        
        # Print the log prior probabilities
        print(f"Log prior probability of Hero: {np.log(len(hero_quotes) / (len(hero_quotes) + len(villain_quotes)))}")
        print(f"Log prior probability of Villain: {np.log(len(villain_quotes) / (len(hero_quotes) + len(villain_quotes)))}")

        # Create the second DataFrame
        df = pd.DataFrame({
            'Category': ['Hero', 'Villain', 'Total'],
            'Number of Quotes': [len(hero_quotes), len(villain_quotes), len(hero_quotes) + len(villain_quotes)],
            'Number of Tokens': [total_hero_words, total_villain_words, total_hero_words + total_villain_words],
            'Number of Types': [len(set(hero_quotes)), len(set(villain_quotes)), total_words],
            'Average Number of Tokens per Document': [avg_hero_tokens, avg_villain_tokens, (total_hero_words + total_villain_words) / (len(hero_quotes) + len(villain_quotes))],
        })

        # Save the DataFrame to a CSV file
        df.to_csv('data_statistics.csv', index=False)

        print(df)
        print()
    
        # theses are for testing the values of the log likelihood ratios
        # Print the log likelihood ratios for the first 5 words in the vocabulary
        for word in vocabulary[:5]:
            print(f"Log likelihood ratio of {word} for Hero: {hero_word_prob[word] - villain_word_prob[word]}")
            print(f"Log likelihood ratio of {word} for Villain: {villain_word_prob[word] - hero_word_prob[word]}")
        print()

        # Print the log likelihood ratios for the last 5 words in the vocabulary
        for word in vocabulary[-5:]:
            print(f"Log likelihood ratio of {word} for Hero: {hero_word_prob[word] - villain_word_prob[word]}")
            print(f"Log likelihood ratio of {word} for Villain: {villain_word_prob[word] - hero_word_prob[word]}")
        print()

        # Print the log likelihood ratios for 5 random words in the vocabulary
        for word in np.random.choice(vocabulary, 5):
            print(f"Log likelihood ratio of {word} for Hero: {hero_word_prob[word] - villain_word_prob[word]}")
            print(f"Log likelihood ratio of {word} for Villain: {villain_word_prob[word] - hero_word_prob[word]}")
        print()
        
        print(bag_of_words_matrix)

       


if __name__ == "__main__":

    main()


