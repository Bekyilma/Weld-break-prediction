#!/usr/bin/python
# coding=utf-8
import sys
import os
import io
import codecs
import pandas as pd
import numpy
import glob
import cPickle

input_path = sys.argv[1]
path = input_path + '/*.txt'
print("Preprocessing dataset from: " + str(path))
files=glob.glob(path)
dataset = []
for file in files:
    df = pd.read_csv(file, sep='\t',
                     header=1, skip_blank_lines=True, skiprows=lambda x:x in range(3,25))
    df2 = df.fillna(0)
    matrix = df2.as_matrix()
    matrix = matrix[:,1:]
    feat_average = numpy.mean(matrix, axis=0)
    dataset.append(feat_average)
cPickle.dump(dataset,open(input_path+'_dataset', "wb"),protocol = cPickle.HIGHEST_PROTOCOL)
