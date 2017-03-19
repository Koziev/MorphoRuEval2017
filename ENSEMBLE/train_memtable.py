# -*- coding: utf-8 -*- 
'''
Статанализ обучающего корпуса для выявления слов, которые
имеют однозначный tagset.
(c) Илья Козиев 2017 для morphrueval_17 inkoziev@gmail.com

Обучение:
python train_memtable.py обучающий_корпус
'''

from __future__ import print_function
import codecs
import collections
import json
import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Masking
from keras.layers.core import Dense, Dropout
from keras.layers import recurrent
import keras.callbacks
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
import os.path

# Приводить ли все к нижнему регистру
LOWERCASE = True

UNAMBIG_FREQ_THRESHOLD = 2

# ----------------------------------------------------------------------

def normalize_word(word):
    return word.lower().replace(u'ё',u'е')

# ----------------------------------------------------------------------

corpus_path = sys.argv[1]
model_folder = sys.argv[-1]

# ----------------------------------------------------------------------

# Загрузим тренировочный датасет.
print( 'Analyzing "{0}..."'.format(corpus_path) )

word2freq = collections.Counter()
word2tagset = dict()
pos2tags = dict()

with codecs.open( corpus_path, 'r', 'utf-8') as rdr:
    sent = []
    for line0 in rdr:
        line = line0.strip()
        if len(line)!=0:
            tx = line.split('\t')
            if len(tx)<2:
                good = False
            else:
                word = normalize_word(tx[1])
                word2freq[word] += 1
                pos = tx[3] # часть речи
                features = tx[4] if len(tx)==5 else u'' # теги
                tagset = (pos + u' ' + features).strip()

                if pos not in pos2tags:
                    pos2tags[pos] = set()

                for tag in features.split(u'|'):
                    if tag!=u'_':
                        tag_name,tag_value = tag.split(u'=')
                        pos2tags[pos].add(tag_name)

                if word in word2tagset:
                    if word2tagset[word]!=tagset:
                        word2tagset[word] = u''
                else:
                    word2tagset[word] = tagset

n1 = sum( ( 1 for w,t in word2tagset.iteritems() if t!=u'' ) )
print( 'done, {0} words with unumbiguous tagset'.format(n1) )

# ------------------------------------------------------------------------

with codecs.open( os.path.join(model_folder,'pos2tags.dat'), 'w', 'utf-8' ) as wrt:
    for pos,tags in pos2tags.iteritems():
        wrt.write( u'{}\t{}\n'.format(pos,unicode.join(u' ',tags)) )

n_stored = 0
with codecs.open( os.path.join(model_folder,'word2tagset.dat'), 'w', 'utf-8' ) as wrt:
    for word,tagset in word2tagset.iteritems():
        if tagset!=u'' and word2freq[word]>=UNAMBIG_FREQ_THRESHOLD:
            wrt.write( u'{}\t{}\n'.format(word,tagset) )
            n_stored += 1

print( '{0} words have been stored'.format(n_stored) )