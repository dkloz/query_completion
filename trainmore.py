import argparse
import logging
import os
import pandas
import numpy as np
import tensorflow as tf

import helper
from dataset import Dataset
from model import Model
from metrics import MovingAvg
from vocab import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--data', type=str, action='append', dest='data',
                    help='where to load the data')
parser.add_argument('--valdata', type=str, action='append', dest='valdata',
                    help='where to load validation data', default=[])
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()

expdir = args.expdir


params = helper.GetParams(None, 'eval', args.expdir)

logging.basicConfig(filename=os.path.join(expdir, 'logfile.more.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

def LoadData(filenames):
  def Prepare(s):
    if type(s) != str:
        print s
    s = str(s)
    return ['<S>'] + list(s) + ['</S>']

  dfs = []
  for filename in filenames:
    df = pandas.read_csv(filename, sep='\t', compression='gzip', header=None)
    df.columns = ['user', 'query_', 'date']
    df['query_'] = df.query_.apply(Prepare)
    df['user'] = df.user.apply(lambda x: 's' + str(x))
    dfs.append(df)
  return pandas.concat(dfs)

df = LoadData(args.data)
char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
user_vocab = Vocab.Load(os.path.join(args.expdir, 'user_vocab.pickle'))
params.user_vocab_size = len(user_vocab)
dataset = Dataset(df, char_vocab, user_vocab, max_len=params.max_len)


model = Model(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
saver.restore(session, os.path.join(expdir, 'model.bin'))
#session.run([model.prev_c.initializer, model.prev_h.initializer])

avg_loss = MovingAvg(0.97)
for idx in range(params.iters):
  feed_dict = dataset.GetFeedDict(model)
  c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
  cc = avg_loss.Update(c)
  if idx % 50 == 0 and idx > 0:
    logging.info({'iter': idx, 'cost': cc, 'rawcost': c})
  if idx % 1000 == 0:
    saver.save(session, os.path.join(expdir, 'model.bin'),
               write_meta_graph=False)
