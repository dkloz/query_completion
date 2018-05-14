import tensorflow as tf
import logging

from model import Model
from metrics import MovingAvg


def train_model(dataset, valdata, params, save_name, emb_matrix):
    tf.reset_default_graph()

    threads = 24
    model = Model(params, learning_rate=0.001)
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads)
    session = tf.Session(config=config)

    session.run(tf.global_variables_initializer())
    session.run(model.emb_init, feed_dict={model.emb_placeholder: emb_matrix})  # init word embeddings

    avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
    total_iterations = max(params.iters, len(dataset)/params.batch_size)
    print 'Doing %d total iterations' % total_iterations
    for idx in range(total_iterations):
        feed_dict = dataset.GetFeedDict(model)
        feed_dict[model.dropout_keep_prob] = params.dropout

        c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
        cc = avg_loss.Update(c)
        if idx % 100 == 0 and idx > 0:
            # test one batch from the validation set
            val_c = session.run(model.avg_loss, valdata.GetFeedDict(model))
            logging.info({'iter': idx, 'cost': cc, 'rawcost': c, 'valcost': val_c})
        if idx % 500 == 0:  # save a model file every 2,000 minibatches
            saver.save(session, save_name, write_meta_graph=False)
