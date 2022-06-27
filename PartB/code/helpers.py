"""
This file contains all functions needed for the Genetic Algorithm.
"""
import os
import pathlib
import random
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

VOCAB_LENGTH = 8520
POPULATION_SIZE = 20
LOWER_BOUND = 1000  # At least 1000 words must be chosen for any solution
DATA_PATH = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "data")

def generate_population(population_size, genome_length):

    def generate_genome(length):

        """
        A genome for the population will be a vector of 8520 ones and zeros.
        If the genome has a 1 at index i, then the corresponding word is chosen for this solution.
        At least LOWER_BOUND must be chosen for any genome (solution).
        This function generates a random genome (containing ones and zeros).
        """

        return random.choices([0, 1], k=length)

    """
    A population is a list of genomes.
    This function generates a random population (containing genomes).
    """

    return [generate_genome(genome_length) for _ in range(population_size)]

def calculate_tf_idf_custom():

    """
    This function calculates the mean of tf-idf metric for every word in the vocab. Custom implementation.
    Calculate once and store in a file.
    """

    with open(os.path.join(DATA_PATH, "train-data.dat"), 'r') as f:
        train_lines = f.readlines()

    N = len(train_lines)
    tf_dict = {}
    document_count_words = [0] * N
    tf_idf_dict = {}
    mean_tf_idf_dict = {}

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

    for i in range(0, VOCAB_LENGTH):
        occurances_in_all_documents = sum(1 if x!=0 else 0 for x in tf_dict[i])
        assert occurances_in_all_documents <= N, "Error: occurances_in_all_documents > N"
        idf = log(N / occurances_in_all_documents)

        for j in range(0, N):
            tf = tf_dict[i][j] / document_count_words[j]
            tf_idf_dict[i][j] = tf * idf
    
    for i in range(0, VOCAB_LENGTH):
        mean_tf_idf_dict[i] = sum(tf_idf_dict[i]) / N
    
    with open(os.path.join(DATA_PATH, "custom_mean_tf_idf_custom.dat"), 'w') as f:
        for i in range(0, VOCAB_LENGTH):
            to_write = str(i) + "," + str(mean_tf_idf_dict[i]) + "\n"
            f.write(to_write)

def calculate_tf_idf_sklearn():

    """
    This function calculates the mean of tf-idf metric for every word in the vocab. Sk-learn implementation.
    Calculate once and store in a file.
    """

    with open(os.path.join(DATA_PATH, "train-data.dat"), 'r') as f:
        train_lines = f.readlines()

    vocab_df = pd.read_csv(os.path.join(DATA_PATH, "vocabs.txt"), sep=',' ,names= ["word", "word_id"], na_filter=False)
    id_to_words = dict(zip(vocab_df.word_id, vocab_df.word))

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
    Genomes with more ones are given a bigger penalty.
    """

    assert len(genome) == VOCAB_LENGTH, "Error: len(genome) != VOCAB_LENGTH"

    num_of_ones = sum(genome)
    if num_of_ones < LOWER_BOUND:
        return 0
    
    genome_score = 0
    for word_id, value in enumerate(genome):
        if value == 1:
            genome_score += tf_idf_mean_dict[word_id]
    
    # The number of ones denotes the percentage of the genome score that will be deducted from the total score. 
    # If a genome has max ones then the penalty will be equal to the genome score and the returned value will be 0.
    penalty = ( ( num_of_ones - LOWER_BOUND ) / ( len(genome) - LOWER_BOUND ) ) * genome_score
    return genome_score - penalty

# GENETIC OPERATORS

def roulette_wheel_pair_selection(population, tf_idf_mean_dict, population_size):

    """
    Select the parents for the next generation.
    """

    # Calculate the fitness of each genome
    fitness_scores = [fitness(genome, tf_idf_mean_dict) for genome in population]
    total_fitness = sum(fitness_scores)

    # Calculate the probability of each genome
    probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]

    parents = random.choices(population, weights=probabilities, k=population_size)

    assert len(parents) == population_size, "Error: len(parents) != population_size"

    return parents

def single_point_crossover(parent1, parent2, pc):

    """
    Perform single point crossover on the parents.
    """

    assert len(parent1) == len(parent2), "Error: len(parent1) != len(parent2)"

    if random.random() < pc:
        cp = random.randint( 1, (len(parent1) - 2) )
        return parent1[:cp] + parent2[cp:], parent2[:cp] + parent1[cp:]

    return parent1, parent2

def mutation(genome, pm):

    """
    Perform mutation on the genome. Flip a bit only if the random number is less than the mutation probability.
    """

    for i in range(0, len(genome)):
        if random.random() < pm:
            genome[i] = 1 - genome[i]

    return genome

if __name__ == "__main__":

    """
    Test the functions.
    """
    # Calculate the mean tf-idf values ONCE for every word and store them in a file
    # calculate_tf_idf_custom()
    # calculate_tf_idf_sklearn()
    # print(single_point_crossover([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], .9))
    # print(mutation([0, 0, 0, 0], .1))

    mean_tf_idf = get_tf_idf_mean("custom_mean_tf_idf_sklearn.dat")
    population = generate_population(20, VOCAB_LENGTH)
    selected_parents = roulette_wheel_pair_selection(population, mean_tf_idf, 20)
    offspring_a , offspring_b = single_point_crossover(selected_parents[0], selected_parents[1], .9)
    offspring_a = mutation(offspring_a, .1)
    offspring_b = mutation(offspring_b, .1)
