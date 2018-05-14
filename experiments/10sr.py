import os
import sys

from experiments.exp_helper import make_data_train_model
from experiments.exp_helper import read_gpu_lr
from experiments.test import test_model

gpu, _ = read_gpu_lr(False)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

dataset_name = '10sr'

#
context = 'user'

exp_name = dataset_name + '_' + context + '_query_10000.txt'
sys.stdout = open(exp_name, 'w')

testdata = make_data_train_model(dataset_name, context)
test_model(dataset_name, context, testdata)
sys.stdout.flush()

context = 'category'

exp_name = dataset_name + '_' + context + '_query_10000.txt'
sys.stdout = open(exp_name, 'w')

testdata = make_data_train_model(dataset_name, context)
test_model(dataset_name, context, testdata)
sys.stdout.flush()
