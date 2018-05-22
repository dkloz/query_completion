import os
import tensorflow as tf
import numpy as np
from model import MetaModel
from util.filenames import expdir
import pandas

threads = 12


def test_model(dataset_name, context, testdata):
    tf.reset_default_graph()
    exp_dir = os.path.join(expdir, dataset_name, context)

    metamodel = MetaModel(exp_dir)
    model_loaded = metamodel.model
    metamodel.MakeSessionAndRestore(threads)

    total_word_count = 0
    total_log_prob = 0
    results = []

    for idx in range(len(testdata.df) / testdata.batch_size):
        feed_dict = testdata.GetFeedDict(model_loaded)
        c, words_in_batch, sentence_costs = metamodel.session.run([model_loaded.avg_loss,
                                                                   model_loaded.words_in_batch,
                                                                   model_loaded.per_sentence_loss], feed_dict)

        total_word_count += words_in_batch
        total_log_prob += float(c * words_in_batch)
        print '{0}\t{1:.3f}'.format(idx, np.exp(total_log_prob / total_word_count))

        lens = feed_dict[model_loaded.query_lengths]
        for length, sentence_cost in zip(lens, sentence_costs):
            data_row = {'length': length, 'cost': sentence_cost}
            results.append(data_row)

    results = pandas.DataFrame(results)
    results.to_csv(os.path.join(exp_dir, 'pplstats.csv'))

    idx = len(testdata.df) / testdata.batch_size
    print '{0}\t{1:.3f}'.format(idx, np.exp(total_log_prob / total_word_count))
