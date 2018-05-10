from util.io import load_array


def load_train(dataset='12m'):
    filename = '/home/dkotzias/query_completion/data/my-aol-data/%s/train.pkl' % dataset
    return load_array(filename)


def load_validation(dataset='12m'):
    filename = '/home/dkotzias/query_completion/data/my-aol-data/%s/val.pkl' % dataset
    return load_array(filename)


def load_test(dataset='12m'):
    filename = '/home/dkotzias/query_completion/data/my-aol-data/%s/test.pkl' % dataset
    return load_array(filename)
