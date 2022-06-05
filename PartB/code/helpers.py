"""
This file contains all functions needed for the Genetic Algorithm.
"""
import os
import pathlib
from math import log
from random import choices, choice

VOCAB_LENGTH = 8520
POPULATION_SIZE = 20
LOWER_BOUND = 1000
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

def repair_procedure(genome):

    """
    This fucntion counts the numbers of ones in the given genome and repairs it.
    It randomly replaces zeros with ones until the genome has LOWER_BOUND ones.
    """

    while sum(genome) < LOWER_BOUND:
        genome[choice( range(0, POPULATION_SIZE) )] = 1

    return genome

def calculate_tf_idf():

    """
    This function calculates the mean of tf-idf metric for every word in the vocab.
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
    with open(os.path.join(DATA_PATH, "mean_tf_idf.dat"), 'w') as f:
        for i in range(0, VOCAB_LENGTH):
            to_write = str(i) + "," + str(mean_tf_idf_dict[i]) + "\n"
            f.write(to_write)

def get_tf_idf_mean():

    """
    Read the file with the mean of the tf idf values and return a dict id: mean tf idf for each word in the Vocabulary.
    """

    tf_idf_mean_dict = {}
    with open(os.path.join(DATA_PATH, "mean_tf_idf.dat"), 'r') as f:
        for line in f.readlines():
            word_id, mean_tf_idf = line.split(",")
            tf_idf_mean_dict[int(word_id)] = float(mean_tf_idf)

    return tf_idf_mean_dict

if __name__ == "__main__":

    # Calculate the mean tf-idf values ONCE for every word and store them in a file
    # calculate_tf_idf()

    mean_tf_idf = get_tf_idf_mean()
    population = generate_population()
    for genome in population:
        genome = repair_procedure(genome)
    print(mean_tf_idf)