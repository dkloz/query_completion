import os
import tensorflow as tf
import numpy as np
from model import MetaModel
from util.filenames import expdir

threads = 12


def test_model(dataset_name, context, testdata):
    tf.reset_default_graph()
    exp_dir = os.path.join(expdir, dataset_name, context)

    metamodel = MetaModel(exp_dir)
    model_loaded = metamodel.model
    metamodel.MakeSessionAndRestore(threads)

    total_word_count = 0
    total_log_prob = 0
    for idx in range(len(testdata.df) / testdata.batch_size):
        feed_dict = testdata.GetFeedDict(model_loaded)
        c, words_in_batch = metamodel.session.run([model_loaded.avg_loss, model_loaded.words_in_batch], feed_dict)

        total_word_count += words_in_batch
        total_log_prob += float(c * words_in_batch)

    idx = len(testdata.df) / testdata.batch_size
    print '{0}\t{1:.3f}'.format(idx, np.exp(total_log_prob / total_word_count))
