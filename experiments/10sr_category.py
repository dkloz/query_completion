import os

from experiments.exp_helper import make_data_train_model
from experiments.exp_helper import read_gpu_lr
from experiments.test import test_model

dataset_name = '10sr'
context = 'category'

gpu, _ = read_gpu_lr(False)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

testdata = make_data_train_model(dataset_name, context)
test_model(dataset_name, context, testdata)
