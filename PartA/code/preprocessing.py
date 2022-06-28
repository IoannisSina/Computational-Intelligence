import os
import pathlib
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "data")
GENETIC_BAG_OF_WORDS = True

def print_sentence(sentence):

    """
    Create a sentence for debugging.
    """

    # Read the training data
    with open(os.path.join(DATA_PATH, "vocabs.txt"), 'r') as f:
        vocab_lines = f.readlines()
    
    vocab = {}
    for line in vocab_lines:
        word, index = line.split(",")
        vocab[int(index.strip())] = word
    
    # Read the training data
    with open(os.path.join(DATA_PATH, "train-data.dat"), 'r') as f:
        train_lines = f.readlines()
    
    if sentence < 1 or sentence > len(train_lines):
        return
    
    to_print_sentence = train_lines[sentence-1]
    for word in to_print_sentence.split():
        if "<" and ">" not in word:
            print(vocab[int(word)], end=" ")

def bag_of_words(data_lines):

    """
    Read the training data and create a bag of words representation.
    The final matrix (based on the input) will be a matrix of size: 8251 x 8520.
    8251 is the number of text docs and 8520 is the number of words in the vocabulary.
    """
    
    data_matrix = np.zeros((len(data_lines), 8520), dtype=int)
    # Create the bag of words representation
    for i, line in enumerate(data_lines):
        for word in line.split():
            if "<" and ">" not in word:
                data_matrix[i][int(word)] += 1
    return data_matrix

def bag_of_words_genetic_vocabs(data_lines):

    """
    Read the training data and create a bag of words representation based on the new vocabs
    """

    with open(os.path.join(DATA_PATH, "new_vocabs_genetic.txt"), 'r') as f:
        new_vocabs_genetic_lines = f.readlines()
    
    new_vocal_len = len(new_vocabs_genetic_lines)
    data_matrix = np.zeros((len(data_lines), new_vocal_len), dtype=int)

    # Create dict old_word_id: [new_word_id]
    old_word_id_to_new_word_id = {}
    for line in new_vocabs_genetic_lines:
        _, old_word_id, new_word_id = line.split(",")
        old_word_id_to_new_word_id[int(old_word_id.strip())] = int(new_word_id.strip())

    for i, line in enumerate(data_lines):
        for word in line.split():
            if "<" and ">" not in word:
                if word in old_word_id_to_new_word_id:
                    data_matrix[old_word_id_to_new_word_id[int(word)]] += 1
    return data_matrix

def normalize_data(data_matrix, choice):

    """
    Normalize the bow matrix. Standardization is used.
    """

    return preprocessing.StandardScaler().fit_transform(data_matrix) if choice else preprocessing.normalize(data_matrix)

def get_dataset(data_file_name, label_file_name):

    """
    Read training or test data and create bag of words representation, and normalize the data (if needed).
    """

    # Read the training or test data
    with open(os.path.join(DATA_PATH, data_file_name), 'r') as f:
        data_lines = f.readlines()
    
    # Read the training or test labels
    with open(os.path.join(DATA_PATH, label_file_name), 'r') as f:
        label_lines = f.readlines()
    
    # Create the bag of words representation and normalize data
    if GENETIC_BAG_OF_WORDS:
        bow_matrix = bag_of_words_genetic_vocabs(data_lines)
    else:
        bow_matrix = bag_of_words(data_lines)

    # print(f"Maximal value in the matrix before norm before normalization: {np.amax(bow_matrix)}")
    # plt.hist(bow_matrix.ravel(), bins=np.arange(np.amin(bow_matrix), np.amax(bow_matrix)),  color='#0504aa', alpha=0.7, rwidth=0.85)
    # plt.yscale('log')
    # plt.margins(x=0.02)
    # plt.show()
    normalized_data = normalize_data(bow_matrix, True)  # True for standardization, False for normalization

    # Create maxtrix for labels
    label_matrix = np.zeros((len(label_lines), 20), dtype=int)
    for i, line in enumerate(label_lines):
        label_matrix[i] = np.array(line.split(), dtype=int)
    
    return normalized_data, label_matrix

if __name__ == "__main__":

    # sentences = [1, 2, 3]
    # for i in sentences:
    #     print_sentence(i)
    #     print("\n")

    X_train, y_train = get_dataset("train-data.dat", "train-label.dat")
    X_test, y_test = get_dataset("test-data.dat", "test-label.dat")
    