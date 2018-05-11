import numpy as np
import os
from util.io import save_pickle
from util.filenames import word_embeddings_dir


def x_to_y(x, end_token=2230):
    """Converts an input matrix x, to an output matrix y, since y is the next word of each sequence.

    Removes the first word (sent_start) and appends at the end the end token <sent_end>

    End token is defined by getting the vocab for '10sr' and doing vocab['<sent_end>']. Can change
    Args:
        x: input matrix of size (sentences, max_sentence_len)
        end_token: id of the end token (<sent_end>).
    Returns:
        Array with labels, of shape (N, max_sent_length, 1)
    """
    y = np.zeros(x.shape)
    y[:, :-1] = x[:, 1:]
    for i in range(y.shape[0]):
        if y[i, -1] != 0:
            y[i, -1] = end_token
        else:
            for j in range(y.shape[1]):
                if y[i, j] == 0:
                    y[i, j] = end_token
                    break
    return np.expand_dims(y, -1)


def context_to_array(context, max_sent_len):
    """Converts a list of context ids to an array ready to be used by an LSTM.
    Essentially the conversion is from a numpy vector of size N, to a numpy array of size (N, max_sent,1).
    Args:
        context: Numpy vector with context for each sequence.
        max_sent_len: Int, length of maximum sequence
    Returns:
        Numpy array (tiled vector) of size (N, max_sent,1)
    """
    return np.repeat(context, max_sent_len).reshape(len(context), max_sent_len)
    # return context.reshape(len(context), max_sent_len, 1)


def get_train_val_mask(length, train_percent):
    """Returns the same mask (for the same length and percent) to maintain random but deterministic datasets.
     Takes in the length and how much should be training and returns a mask for training and validation.
     Args:
         length: integer, size of total data points.
         train_percent: float: How much data should be in training (rest are in validation).
     Returns:
         List of elements that will be in training, List of elements that belong in validation.
     """
    np.random.seed(12345)
    mask = np.random.permutation(length)
    train_lim = int(length * train_percent)
    train_mask = mask[:train_lim]
    val_mask = mask[train_lim:]
    return train_mask, val_mask


def split_train_val(train_percent, sentences, end_token=2230):
    """Splits in train and val and Prepares a set of sentences to be used in an LSTM.
    Shuffles data
    Splits in training and validation
    Creates target values for train and validation.
    Args:
        train_percent: float: How much data should be in training (rest are in validation).
        sentences: Data (array of Nx max_sent_length)
    Returns:
        training array of shape N*train_percent, sentence length, training labels, (Same size -1)
        validation array of shape N*train_percent, sentence length, validation labels, (Same size -1)
    """
    train_mask, val_mask = get_train_val_mask(len(sentences), train_percent)

    x_train = sentences[train_mask]
    y_train = x_to_y(sentences[train_mask], end_token)

    x_val = sentences[val_mask]
    y_val = x_to_y(sentences[val_mask], end_token)

    return x_train, y_train, x_val, y_val


def split_train_val_context(train_percent, sentences, context=None, end_token=2230):
    """Splits data in training and validation and splits context in the same way.
    Also repeats and reshapes context to be in the form that an LSTM requires

    Args:
        train_percent: float: How much data should be in training (rest are in validation).
        sentences: Data (array of Nx max_sent_length)
        context: Array of context. 1D that is converted
        end_token: id of the ending sentence token.
    Returns:
        train data (list of sentences, context), train labels, val data, val labels.
    """

    train_mask, val_mask = get_train_val_mask(len(sentences), train_percent)
    x_train, y_train, x_val, y_val = split_train_val(train_percent, sentences, end_token)
    max_sent_length = x_train.shape[1]

    x_train = [x_train]
    x_val = [x_val]
    train_zero_mask = np.where(x_train[0] == 0)
    val_zero_mask = np.where(x_val[0] == 0)

    for c in context:
        c_train = context_to_array(c[train_mask], max_sent_length)
        c_val = context_to_array(c[val_mask], max_sent_length)

        c_train[train_zero_mask] = 0
        c_val[val_zero_mask] = 0

        x_train.append(c_train)
        x_val.append(c_val)

    return x_train, y_train, x_val, y_val


def sample_data(x_matrices, y, size=10000, replace=False, seed=1234):
    """Methods that returns a random sample from data.

    Setting a seed so that the sample is random, but deterministic.
    Args:
        x_matrices: List of words or context arrays. Can have one or more contexts. Each has shape (None, 35)
        y: Array of shape (None, 35, 1) for target word id.
        size: Sample size.
        replace: Determines whether to sample with replacement or not.
        seed: The seed to be set. Change that to time to make truly random
    Returns:
        x_val_sample, y_val_sample in the same shapes as input.
    """
    np.random.seed(seed)
    mask = np.random.choice(len(y), size, replace=replace)
    x_sample = []
    for inp in x_matrices:
        x_sample.append(inp[mask])

    y_sample = y[mask]
    return x_sample, y_sample


def load_word_embeddings_from_file(name='glove', size=300, vocab=None):
    """Loads the file from glove that contains word embedding vectors. File should exists!"""
    filename = '%s.840B.%dd.txt' % (name, size)
    filename = os.path.join(word_embeddings_dir, filename)
    print filename
    emb_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            if vocab is not None:
                if vocab.get(word, -1) == -1:
                    continue

            emb = [float(t) for t in tokens[1:]]
            emb_dict[word] = emb
    return emb_dict


def print_intersection(vocab, glove_emb):
    """Prints intersection between two dictionaries (their keys)."""
    v_keys = set(vocab.keys())
    g_keys = set(glove_emb.keys())
    intersection = v_keys.intersection(g_keys)
    inter_percent = (1. * len(intersection)) / len(vocab)
    print 'Total words intersected %d which is %.2f of the vocab' % (len(intersection), inter_percent)


def make_word_embedding_matrix_file(vocab, dataset_name='9sr'):
    """Makes a word embedding matrix file, ready to be loaded as the weights of an embedding layer in a NN.
    Reads the vocabulary of a dataset as returned by get_vocab. (This must exist).
    Saves the word embedding matrix in the word_embeddings_dir with appropriate name.

    Args:
        vocab: vocabulary for that dataset. word -> id
        dataset_name: String that defines the dataset name and location of vocab.
    """
    np.random.seed(12345)  # random but reproducable
    glove_emb = load_word_embeddings_from_file(vocab=vocab)

    print_intersection(vocab, glove_emb)

    # word_emb_array = np.random.random((len(vocab) + 1, 300)) # +1 is only for my code
    word_emb_array = np.random.random((len(vocab), 300))
    for w, i in vocab.items():
        if glove_emb.get(w) is not None:
            word_emb_array[i] = np.array(glove_emb.get(w))

    filename = os.path.join(word_embeddings_dir, '%s_glove_matrix_300d.pkl' % dataset_name)
    save_pickle(filename, word_emb_array)

    return word_emb_array, glove_emb
