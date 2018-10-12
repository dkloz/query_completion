"""A file that contains helpful IO wrapper functions.
Most of them have a verbose options (print what is being saved or loaded. They also have an option to change
the permission of the files being created. (This is handy for creating files on the servers)."""
import cPickle as pickle
import time
import numpy as np
import sys
import json
import os
# import unicodecsv as csv
import stat
# import pandas as pd

GO_RW_PERM = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRWXG | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH


def make_go_rw(filename, change_perm=True):
    if change_perm:
        os.chmod(filename, 0770)


def file_exists(filename):
    return os.path.isfile(filename)


def make_dir(filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def build_path(dir, filename):
    return os.path.join(dir, filename)


def extract_filename(path):
    return os.path.split(path)[1]


def save_pickle(filename, obj, verbose=False, other_permission=True):
    make_dir(filename)
    if verbose:
        print '--> Saving ', filename, ' with pickle was ',
        sys.stdout.flush()
    t = time.time()
    with open(filename, 'wb') as gfp:
        pickle.dump(obj, gfp, protocol=pickle.HIGHEST_PROTOCOL)
        gfp.close()

    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)


def save_array(filename, obj, verbose=True, other_permission=True):
    filename = filename.replace('.pkl', 'npy')
    make_dir(filename)
    if verbose:
        print '--> Saving ', filename, ' with np.array was ',
    sys.stdout.flush()
    t = time.time()
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj)
    np.save(filename, obj)
    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)


def save_arrayz(filename, obj, verbose=False, other_permission=True):
    filename = filename.replace('.pkl', 'npz')
    make_dir(filename)
    if verbose:
        print '--> Saving Z', filename, ' with np.array was ',
    sys.stdout.flush()
    t = time.time()
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj)
    np.savez(filename, obj)
    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)


def save_txt(filename, obj, delimiter=',', fmt='% .4e', verbose=True, other_permission=True):
    make_dir(filename)
    if verbose:
        print '--> Saving ', filename, ' with np.savetxt was ',
    sys.stdout.flush()
    t = time.time()
    np.savetxt(filename, obj, delimiter=delimiter, fmt=fmt)
    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)


def save_text_sentences(filename, sentences, delimiter='\n', verbose=True, other_permission=True):
    make_dir(filename)
    t = time.time()
    if verbose:
        print '--> Saving ', filename, ' as a text file was ',
    sys.stdout.flush()
    with open(filename, 'w') as f:
        for s in sentences:
            f.write('%s%s' % (s, delimiter))
        f.close()
    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)


def save_json(filename, lil, verbose=True, other_permission=True):
    make_dir(filename)
    if verbose:
        print '--> Saving ', filename, ' with json was ',
    sys.stdout.flush()
    t = time.time()
    with open(filename, 'w') as fp:
        json.dump(lil, fp)

    fp.close()
    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)

#
# def save_csv(filename, obj, verbose=True, other_permission=True):
#     """
#     saves a list of lists as a csv file
#     :param filename: str
#     :param obj: list of lists
#     :param verbose: print time or not
#     :param other_permission: boolean add read and write for group and others
#     :return:
#     """
#     make_dir(filename)
#     if verbose:
#         print '--> Saving ', filename, ' with csv writer was ',
#         sys.stdout.flush()
#
#     t = time.time()
#     with open(filename, "w") as f:
#         writer = csv.writer(f)
#         for row in obj:
#             writer.writerow(row)
#
#     if verbose:
#         print '%.3f s' % (time.time() - t)
#
#     make_go_rw(filename, other_permission)


def load_pickle(filename, verbose=False):
    if verbose:
        print '--> Loading ', filename, ' with pickle was ',
        sys.stdout.flush()
    t = time.time()
    with open(filename, 'rb') as gfp:
        r = pickle.load(gfp)

    if verbose:
        print '%.3f s' % (time.time() - t)
    return r


def load_array(filename, verbose=True):
    if verbose:
        print '--> Loading ', filename, ' with np.load was ',
    filename = filename.replace('.pkl', 'npz')  # in case it exists
    sys.stdout.flush()
    t = time.time()
    r = np.load(filename)
    if verbose:
        print '%.3f s' % (time.time() - t)
    return r


def load_json(filename):
    print '--> Loading ', filename, ' with json was ',
    sys.stdout.flush()
    t = time.time()
    with open(filename, 'r') as fp:
        data = json.load(fp)

    print '%.3f s' % (time.time() - t)
    return data


def load_txt(filename, delimiter=',', verbose=True):
    if verbose:
        print '--> Loading ', filename, ' with np.loadtxt was ',
    sys.stdout.flush()
    t = time.time()
    d = np.loadtxt(filename, delimiter=delimiter)
    if verbose:
        print '%.3f s' % (time.time() - t)
    return d


# def load_csv(filename):
#     print '--> Loading ', filename, ' with np.loadtxt was ',
#     sys.stdout.flush()
#
#     t = time.time()
#     with open(filename, 'r') as fp:
#         reader = csv.reader(fp)
#         data = [row for row in reader]
#
#     print '%.3f s' % (time.time() - t)
#     return data


def load_gen_from_text(filename, delimiter=','):
    print '--> Loading ', filename, ' with np.loadtxt was ',
    sys.stdout.flush()

    t = time.time()
    data = np.genfromtxt(filename, delimiter=delimiter)

    print '%.3f s' % (time.time() - t)
    return data


# def save_datafram(filename, df, verbose=True):
#     if verbose:
#         print 'Saving  %s' % filename
#     make_dir(filename)
#
#     start = time.time()
#     df.to_pickle(filename)
#     os.chmod(filename, 0770)
#
#     if verbose:
#         print 'Saving took %d seconds' % (time.time() - start)
#
#
# def load_dataframe(filename, verbose=True):
#     if verbose:
#         print 'Loading %s' % filename
#     start = time.time()
#     data = pd.read_pickle(filename)
#     if verbose:
#         print 'Loading took %d seconds' % (time.time() - start)
#
#     return data
