import os
import json

from util.filenames import q_data_dir, expdir, word_embeddings_dir
from util.data_io import get_word_embeddings
from util.data_util import make_word_embedding_matrix_file
from dataset import MyDataset
from experiments.train import train_model

from dataset import myLoadData
from vocab import Vocab
from helper import GetParams

max_len = 36
batch_size = 150
num_units = 1024
iterations = 20000
word_embed_size = 300


def read_gpu_lr(read_lr=True):
    lr = 0.0
    gpu = 0
    while read_lr:
        print 'Common lr 7e-4, 10e-4, 15e-4, 20e-4, etc'
        try:
            lr = float(raw_input('Please provide a learning rate: '))
            read_lr = False
        except ValueError:
            print "Not a floating point number"

    done = False
    while not done:
        try:
            gpu = int(raw_input('Please provide GPU id to run this on: '))
            done = True
        except ValueError:
            print "Not a number"
            done = False

    print 'Learning rate set to %f' % lr
    print 'Running on GPU %d' % gpu
    return gpu, lr


def get_emb_size(dataset, context):
    if dataset == '10sr':
        if 'user' in context:
            return 30
        return 6

    if dataset == '200sr':
        if 'user' in context:
            return 50
        return 13

    if dataset == 'yelp':
        if 'user' in context:
            return 11
        return 9


def get_data(dataset_name, context):
    train_filenames = [os.path.join(q_data_dir, dataset_name, 'all_%s.txt' % context)]
    train_df = myLoadData(train_filenames)  # loads data in data frame (pandas)

    val_filenames = [os.path.join(q_data_dir, dataset_name, 'all_%s_val.txt' % context)]
    val_df = myLoadData(val_filenames)  # loads data in data frame (pandas)

    test_filenames = [os.path.join(q_data_dir, dataset_name, 'all_%s_test.txt' % context)]
    test_df = myLoadData(test_filenames)  # loads data in data frame (pandas)

    return train_df, val_df, test_df


def make_vocabs(train_df, val_df):
    # Make vocabs
    word_vocab = Vocab.MakeFromData(train_df.sentence.tolist() + val_df.sentence.tolist(), min_count=1,
                                    no_special_syms=False)
    word_vocab.Save(os.path.join(expdir, 'char_vocab.pickle'))

    category_vocab = Vocab.MakeFromData([[u] for u in train_df.user], min_count=1, no_special_syms=False)
    category_vocab.Save(os.path.join(expdir, 'user_vocab.pickle'))

    return word_vocab, category_vocab


def get_emb_matrix(word_vocab, dataset_name):
    we_filename = os.path.join(word_embeddings_dir, 'query_completion', '%s_glove_matrix_300d.pkl' % dataset_name)
    print we_filename
    if os.path.exists(we_filename):
        emb_matrix = get_word_embeddings(dataset_name)
    else:
        print 'Making word embeddings'
        emb_matrix, emb_dict = make_word_embedding_matrix_file(word_vocab.word_to_idx, dataset_name)

    print emb_matrix.shape
    assert emb_matrix.shape[0] == len(word_vocab)


def get_save_params(dataset_name, context, word_vocab, category_vocab, context_emb_size):
    user_embed_size = context_emb_size

    rank = context_emb_size  # set the rank the same size as context. May get better results if not, but for saving tim

    exp_dir = os.path.join(expdir, dataset_name, context)
    params_filename = '/home/dkotzias/query_completion/default_params.json'
    params = GetParams(params_filename, 'train', exp_dir)

    params.vocab_size = len(word_vocab)
    params.user_vocab_size = len(category_vocab)
    params.max_len = max_len
    params.batch_size = batch_size
    params.num_units = num_units
    params.user_embed_size = user_embed_size
    params.char_embed_size = word_embed_size
    params.iters = iterations
    params.rank = rank

    # save the params
    param_filename = os.path.join(exp_dir, 'params.json')
    param_dict = params.toDict()
    with open(param_filename, 'w') as f:
        json.dump(param_dict, f)
    return params


def make_data_train_model(dataset_name, context):
    context_emb_size = get_emb_size(dataset_name, context)

    train_df, val_df, test_df = get_data(dataset_name, context)
    word_vocab, category_vocab = make_vocabs(train_df, val_df)

    dataset = MyDataset(train_df, word_vocab, category_vocab, max_len=max_len, batch_size=batch_size)
    valdata = MyDataset(val_df, word_vocab, category_vocab, max_len=max_len, batch_size=batch_size)
    testdata = MyDataset(test_df, word_vocab, category_vocab, max_len=max_len, batch_size=batch_size)

    emb_matrix = get_emb_matrix(word_vocab, dataset_name)
    params = get_save_params(dataset_name, context, word_vocab, category_vocab, context_emb_size)

    save_name = os.path.join(expdir, dataset_name, context, 'model.bin')
    train_model(dataset, valdata, params, save_name, emb_matrix)
    return testdata
