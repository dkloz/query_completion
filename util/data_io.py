from util.io import load_array, load_pickle
import os

from util.filenames import word_embeddings_dir


def load_train(dataset='12m'):
    filename = '/home/dkotzias/query_completion/data/my-aol-data/%s/train.pkl' % dataset
    return load_array(filename)


def load_validation(dataset='12m'):
    filename = '/home/dkotzias/query_completion/data/my-aol-data/%s/val.pkl' % dataset
    return load_array(filename)


def load_test(dataset='12m'):
    filename = '/home/dkotzias/query_completion/data/my-aol-data/%s/test.pkl' % dataset
    return load_array(filename)


def get_word_embeddings(dataset='9sr'):
    """Returns 300d embedding matrix based on 50k vocab from first_all."""
    if '10sr' in dataset or 'sr10' in dataset or 'sr9' in dataset or '9sr' in dataset or '200sr' in dataset:
        dataset = '9sr'
    filename = os.path.join(word_embeddings_dir, 'query_completion', '%s_glove_matrix_300d.pkl' % dataset)
    if os.path.exists(filename):
        return load_pickle(filename, False)
    else:
        print ' I do not have word embeddings for %s. Making it now... Takes a few minutes.' % dataset
