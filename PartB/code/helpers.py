"""
This file contains all functions needed for the Genetic Algorithm.
"""

import os
import pathlib
import random
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np

VOCAB_LENGTH = 8520
LOWER_BOUND = 1000  # At least 1000 words must be chosen for any solution
DATA_PATH = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "data")
DATA_PATH_plots = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "plots")

def generate_population(population_size=20, genome_length=VOCAB_LENGTH):

    def generate_genome(length):

        """
        A genome for the population will be a vector of 8520 ones and zeros.
        If the genome has a 1 at index i, then the corresponding word is chosen for this solution.
        At least LOWER_BOUND must be chosen for any genome (solution).
        This function generates a random genome (containing ones and zeros).
        """

        return np.random.randint(0, 2, length).tolist()

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
    for i in range(VOCAB_LENGTH):
        tf_dict[i] = [0] * N
        tf_idf_dict[i] = [0] * N

    # Read all documents and fill all lists
    for i, line in enumerate(train_lines):
        for word in line.split():
            if "<" and ">" not in word:
                tf_dict[int(word)][i] += 1  # Count the number of occurances of the word in the document
                document_count_words[i] += 1  # Count the number of words in the document

    for i in range(VOCAB_LENGTH):
        occurances_in_all_documents = sum(1 if x!=0 else 0 for x in tf_dict[i])
        assert occurances_in_all_documents <= N, "Error: occurances_in_all_documents > N"
        idf = log(N / occurances_in_all_documents)

        for j in range(N):
            tf = tf_dict[i][j] / document_count_words[j]
            tf_idf_dict[i][j] = tf * idf
    
    for i in range(VOCAB_LENGTH):
        mean_tf_idf_dict[i] = sum(tf_idf_dict[i]) / N
    
    with open(os.path.join(DATA_PATH, "custom_mean_tf_idf.dat"), 'w') as f:
        for i in range(VOCAB_LENGTH):
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

    with open(os.path.join(DATA_PATH, "sklearn_mean_tf_idf.dat"), 'w') as f:
        for i in range(VOCAB_LENGTH):
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
    
    score = 0
    for word_id, value in enumerate(genome):
        if value == 1:
            score += tf_idf_mean_dict[word_id]
    
    # The number of ones denotes the percentage of the genome score that will be deducted from the total score.
    # If a genome has max ones then the penalty will be equal to the genome score and the returned value will be 0.
    penalty = ( ( num_of_ones - LOWER_BOUND ) / ( len(genome) - LOWER_BOUND ) ) * score
    return score - penalty

# GENETIC OPERATORS

def roulette_wheel_pair_selection(population, tf_idf_mean_dict):

    """
    Select the parents for the next generation.
    Not used here because the genomes have very similar fitness values.
    """

    initial_size = len(population)

    # Calculate the fitness of each genome
    fitness_scores = [fitness(genome, tf_idf_mean_dict) for genome in population]
    total_fitness = sum(fitness_scores)

    # Calculate the probability of each genome
    probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]

    parents = random.choices(population, weights=probabilities, k=initial_size)

    assert len(parents) == initial_size, "Error: len(parents) roulette != population_size"

    return parents

def tournament_selection(population, tf_idf_mean_dict, k=10):

    """
    Select the parents for the next generation.
    """

    initial_size = len(population)

    all_fitness = [fitness(genome, tf_idf_mean_dict) for genome in population]

    parents = []
    for _ in range(initial_size):
        random_pos = np.random.randint(initial_size)

        for temp_random_pos in np.random.randint(0, initial_size, k):
            if all_fitness[temp_random_pos] > all_fitness[random_pos]:
                random_pos = temp_random_pos
        
        parents.append(population[random_pos])

    assert len(parents) == initial_size, "Error: len(parents) tournament != population_size"

    return parents

def single_point_crossover(parent1, parent2, pc=.9):

    """
    Perform single point crossover on the parents.
    """

    assert len(parent1) == len(parent2), "Error: len(parent1) != len(parent2)"

    if np.random.rand() < pc:
        cp = np.random.randint(1, len(parent1)-2)
        return parent1[:cp] + parent2[cp:], parent2[:cp] + parent1[cp:]

    return parent1, parent2

def mutation(genome, pm=.1):

    """
    Perform mutation on the genome. Flip a bit only if the random number is less than the mutation probability.
    """

    for i in range(len(genome)):
        if np.random.rand() < pm:
            genome[i] = 1 - genome[i]

    return genome

# Plot function

def plot_fitness(fitness_history, generations, population_size, pc, pm):

    """
    Plot the fitness history.
    """

    x = list(range(generations))
    plt.plot(x, fitness_history, color='red')
    plt.xlabel("Generations")
    plt.ylabel("Mean Fitness")
    plt.title("Fitness History | Pop size: {} | PC: {} | PM: {} | Gens: {}".format(population_size,  round(pc,4), round(pm,4),  generations+1))
    filename = "fitness_pop_size_{}_pc_{}_pm_{}.png".format(population_size, round(pc,4), round(pm,4))
    plt.savefig(os.path.join(DATA_PATH_plots, filename))
    plt.close()

if __name__ == "__main__":

    """
    Test the functions.
    """
    # Calculate the mean tf-idf values ONCE for every word and store them in a file
    # calculate_tf_idf_sklearn()
    # print(single_point_crossover([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], .9))
    # print(mutation([0, 0, 0, 0], .1))

    mean_tf_idf = get_tf_idf_mean("sklearn_mean_tf_idf.dat")
    population = generate_population()
    selected_parents_roulette = roulette_wheel_pair_selection(population, mean_tf_idf)
    selected_parents_tournament = tournament_selection(population, mean_tf_idf)
    offspring_a , offspring_b = single_point_crossover(selected_parents_tournament[0], selected_parents_tournament[1])
    offspring_a = mutation(offspring_a)
    offspring_b = mutation(offspring_b)
    