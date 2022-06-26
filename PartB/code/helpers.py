"""
This file contains all functions needed for the Genetic Algorithm.
"""
import os
import pathlib
from math import log
from random import choices
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

VOCAB_LENGTH = 8520
POPULATION_SIZE = 20
LOWER_BOUND = 1000  # At least 1000 words must be chosen for any solution
DATA_PATH = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "data")

def generate_genome(length: int):

    """
    A genome for the population will be a vector of 8520 ones and zeros.
    If the genome has a 1 at index i, then the corresponding word is chosen for this solution.
    At least LOWER_BOUND must be chosen for any genome (solution).
    This function generates a random genome (containing ones and zeros).
    """

    return choices([0, 1], k=length)

def generate_population(size=POPULATION_SIZE, genome_length=VOCAB_LENGTH):

    """
    A population is a list of genomes.
    This function generates a random population (containing genomes).
    """

    return [generate_genome(genome_length) for _ in range(size)]

def calculate_tf_idf_custom():

    """
    This function calculates the mean of tf-idf metric for every word in the vocab. Custom implementation.
    Calculate once and store in a file.
    """

    # Read the training data
    with open(os.path.join(DATA_PATH, "train-data.dat"), 'r') as f:
        train_lines = f.readlines()

    N = len(train_lines)  # Number of training documents
    tf_dict = {}  # Dictionary of tf values for every word
    document_count_words = [0] * N  # Number of words in each document
    tf_idf_dict = {} # Dictionary of tf-idf values for every word in each document
    mean_tf_idf_dict = {} # Dictionary of mean tf-idf values for every word

    # Fill the idf_dict with zeros lists. For every word count the number of occurances in each document. If the word is not in the document, it is 0
    for i in range(0, VOCAB_LENGTH):
        tf_dict[i] = [0] * N
        tf_idf_dict[i] = [0] * N

    # Read all documents and fill all lists
    for i, line in enumerate(train_lines):
        for word in line.split():
            if "<" and ">" not in word:
                tf_dict[int(word)][i] += 1  # Count the number of occurances of the word in the document
                document_count_words[i] += 1  # Count the number of words in the document

    # Fill tf-idf dictionary with the tf-idf values
    for i in range(0, VOCAB_LENGTH):
        occurances_in_all_documents = sum(1 if x!=0 else 0 for x in tf_dict[i])  # Used for idf calculation
        assert occurances_in_all_documents <= N, "Error: occurances_in_all_documents > N"
        idf = log(N / occurances_in_all_documents)  # Calculate idf once

        for j in range(0, N):
            tf = tf_dict[i][j] / document_count_words[j]  # Calculate tf
            tf_idf_dict[i][j] = tf * idf  # Calculate tf-idf
    
    # Calculate the mean tf-idf for every word
    for i in range(0, VOCAB_LENGTH):
        mean_tf_idf_dict[i] = sum(tf_idf_dict[i]) / N
    
    # Store the mean tf-idf values in a file
    with open(os.path.join(DATA_PATH, "custom_mean_tf_idf_custom.dat"), 'w') as f:
        for i in range(0, VOCAB_LENGTH):
            to_write = str(i) + "," + str(mean_tf_idf_dict[i]) + "\n"
            f.write(to_write)

def calculate_tf_idf_sklearn():

    """
    This function calculates the mean of tf-idf metric for every word in the vocab. Sk-learn implementation.
    Calculate once and store in a file.
    """

    # Read the training data
    with open(os.path.join(DATA_PATH, "train-data.dat"), 'r') as f:
        train_lines = f.readlines()

    # Read the vocabulary
    vocab_df = pd.read_csv(os.path.join(DATA_PATH, "vocabs.txt"), sep=',' ,names= ["word", "word_id"], na_filter=False)
    id_to_words = dict(zip(vocab_df.word_id, vocab_df.word))

    # Create corpus and vectorizer
    corpus = []
    for line in train_lines:
        doc = ""
        for word_id in line.split():
            if "<" and ">" not in word_id:
                doc += id_to_words[int(word_id)] + " "
        corpus.append(doc.rstrip())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    assert X.shape[0] == len(corpus), "Error: X.shape[0] != len(corpus)"
    assert X.shape[1] == VOCAB_LENGTH, "Error: len(vectorizer.get_feature_names()) != VOCAB_LENGTH"
    
    tf_idf_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

    # Store the mean tf-idf values in a file
    with open(os.path.join(DATA_PATH, "custom_mean_tf_idf_sklearn.dat"), 'w') as f:
        for i in range(0, VOCAB_LENGTH):
            word = id_to_words[i]
            to_write = str(i) + "," + str(tf_idf_df[word].mean()) + "\n"
            f.write(to_write)

def get_tf_idf_mean(filename):

    """
    Read the file with the mean of the tf idf values and return a dict id: mean tf idf for each word in the Vocabulary.
    """

    tf_idf_mean_dict = {}
    with open(os.path.join(DATA_PATH, filename), 'r') as f:
        for line in f.readlines():
            word_id, mean_tf_idf = line.split(",")
            tf_idf_mean_dict[int(word_id)] = float(mean_tf_idf)

    return tf_idf_mean_dict

def fitness(genome, tf_idf_mean_dict):

    """
    Calculate the fitness of the given genome. If the genome has less than LOWER_BOUND ones, return 0.
    The score is calculated based on the tf-idf values and the number of ones in the genome.
    Genomes with more ones are given a penalty.
    """

    if sum(genome) < LOWER_BOUND:
        return 0
    
    genome_score = 0 
    for word_id, value in enumerate(genome):
        if value == 1:
            genome_score += tf_idf_mean_dict[word_id]

    return genome_score

if __name__ == "__main__":

    # Calculate the mean tf-idf values ONCE for every word and store them in a file
    # calculate_tf_idf_custom()
    calculate_tf_idf_sklearn()

    tf_idf_filename = "custom_mean_tf_idf_sklearn.dat"
    mean_tf_idf = get_tf_idf_mean(tf_idf_filename)
   
    # mean_tf_idf = get_tf_idf_mean(tf_idf_filename)
    # population = generate_population()
