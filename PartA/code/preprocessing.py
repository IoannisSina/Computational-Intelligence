import os
import pathlib
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "data")

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

def bag_of_words():

    """
    Read the training data and create a bag of words representation.
    The final matrix (based on the input) will be a matrix of size: 8251 x 8520.
    8251 is the number of text docs and 8520 is the number of words in the vocabulary.
    """

    # Read the training data
    with open(os.path.join(DATA_PATH, "train-data.dat"), 'r') as f:
        lines = f.readlines()
    
    matrix = np.zeros((len(lines), 8520), dtype=int)
    # Create the bag of words representation
    for i, line in enumerate(lines):
        for word in line.split():
            if "<" and ">" not in word:
                matrix[i][int(word)] += 1
    return matrix

if __name__ == "__main__":

    # sentences = [1, 2, 3]
    # for i in sentences:
    #     print_sentence(i)
    #     print("\n")
    bag_of_words()
