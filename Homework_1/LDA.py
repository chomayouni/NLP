# you will need to use pip to install the following packages: nltk, tqdm, numpy, scikit-learn
import re
import nltk
import tqdm
import argparse
import string
import numpy as np
import sys
import math
import gensim
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from pathlib import Path
from gensim.models import LdaModel
from gensim.corpora import Dictionary

from wordcloud import WordCloud

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

def setup_lda(documents, num_topics):
    # I was having a tone of difficulty with the LDA model, so I used the following resources to help me:
    # https://www.youtube.com/watch?v=Y79sCtzddyA

    # Create a dictionary from the documents
    # this is essentially a list that contains the number of times a word appears in the training set
    documents = [word_tokenize(doc) for doc in documents]
    id2word = Dictionary(documents)
    count = 0
    for k, v in list(id2word.items()):
        print(k, v, dict(id2word.dfs)[k]) #.dfs is the document frequency for each word
        count += 1
        if count > 10:
            break
    # Filter out words that occur less than 5 documents, or more than 50% of the documents
    id2word.filter_extremes(no_below=2, no_above=0.5)

    # Convert documents into a document-term matrix
    corpus_lda_bow = [id2word.doc2bow(doc) for doc in documents]

    last_index = len(id2word) - 1
    print(id2word[last_index])

    for i in range(len(corpus_lda_bow)):
        for word_id, freq in corpus_lda_bow[i]:
            print("word {} (\"{}\") appears {} time.".format(word_id, id2word[word_id], freq))
    bow = corpus_lda_bow[last_index]

     # Build the LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus_lda_bow, id2word=id2word, num_topics=num_topics, passes=2, workers=2)
    
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
    print()

    for index, score in sorted(lda_model[bow], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    print()

    # Calculate grid size
    grid_size = math.ceil(math.sqrt(lda_model.num_topics))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    # Generate word clouds for each topic
    for t in range(lda_model.num_topics):
        ax = axs[t//grid_size, t%grid_size]
        ax.imshow(WordCloud().fit_words(dict(lda_model.show_topic(t, 200))))
        ax.axis("off")
        ax.set_title("Topic #" + str(t))

    # Remove unused subplots
    if lda_model.num_topics < grid_size * grid_size:
        for t in range(lda_model.num_topics, grid_size * grid_size):
            fig.delaxes(axs.flatten()[t])

    plt.tight_layout()
    plt.show()
    import pandas as pd
    import seaborn as sns

    # Create a DataFrame with zeros
    df = pd.DataFrame(0, index=range(len(corpus_lda_bow)), columns=[id2word[i] for i in range(len(id2word))])

    # Fill the DataFrame with word frequencies
    for doc_id, doc in enumerate(corpus_lda_bow):
        for word_id, freq in doc:
            df.at[doc_id, id2word[word_id]] = freq

    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, cmap="YlGnBu")
    plt.show()


    # Number of words to display per topic
    num_words = 10

    # Calculate the number of rows and columns for the subplots
    num_topics = lda_model.num_topics
    cols = int(np.sqrt(num_topics))
    rows = num_topics // cols 
    rows += num_topics % cols

    # Create a new figure for the plots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    # Ensure axes is a 2-dimensional array
    if rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    # For each topic
    for idx, topic in enumerate(lda_model.print_topics(-1)):
        # Get the top words in the topic
        top_words = lda_model.show_topic(idx, num_words)
        
        # Separate the words and their weights into two lists
        words, weights = zip(*top_words)
        
        # Get the current axis
        ax = axes[idx // cols, idx % cols]
        
        # Create a bar chart on the current axis
        ax.barh(range(num_words), weights, tick_label=words)
        
        # Invert the y-axis so the words are displayed top-to-bottom
        ax.invert_yaxis()
        
        # Add a title
        ax.set_title(f"Topic {idx}")

    # Remove unused subplots
    if num_topics < rows * cols:
        for idx in range(num_topics, rows * cols):
            fig.delaxes(axes.flatten()[idx])

    # Show the plots
    plt.tight_layout()
    plt.show()


    return lda_model, corpus_lda_bow, id2word
    

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


        #----------------LDAModel----------------
        # Define the number of topics and the specific words
        # Combine hero and villain quotes into one list of documents
        # documents = [nltk.word_tokenize(quote) for quote in hero_quotes + villain_quotes]
        documents = hero_quotes + villain_quotes
        # Set up the LDA model
        lda_model, corpus, id2word = setup_lda(documents, num_topics=2)
        # Print the topics
        topics = lda_model.print_topics()
        for topic in tqdm.tqdm(topics, desc="printing topics"):
            print(topic)
        

if __name__ == "__main__":

    main()


