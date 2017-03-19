# -*- coding: utf-8 -*- 
'''
POS Tagger на базе символьной сеточной модели
Второй вариант - отдельные классификаторы для части речи и для каждого значимого тега
(c) Илья Козиев 2017 для morphrueval_17 inkoziev@gmail.com

Обучение:
python train_chars_postagger(2).py обучающий_корпус

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
import unicodedata
import os.path

# Полный размер окна (распознаваемое слово входит в него)
WINDOW=11

# Цепочка символов в каждом слове рассматривается в прямом и в обратном порядке,
# облегчая для нейросети работу как с корнями, так и с окончаниями слов.
BIDIR = True

# Цепочка символов в каждом слове рассматривается в обратном порядке (справа налево)
INVERT = False

# ограничение на число сэмплов в датасете - для удобства отладки
max_patterns = 1000000

# ----------------------------------------------------------------------

salient_pos = { u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET' }


def classifiable_pos(part_of_speech):
    return True if part_of_speech in salient_pos else False


def normalize_word( word ):
    return word.lower().replace(u'ё',u'е')


valid_chars = set( [  c for c in u'abcdefghijklmnopqrstuvwxyz0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюя' ] )

def normalize_char(c):
    if c in valid_chars:
        return c
    elif unicodedata.category(c) == 'Po':
        return c
    else:
        return u' '


def word_chars( word, invert ):
    return [ normalize_char(c) for c in (word[::-1] if invert else word) ]

# ----------------------------------------------------------------------

def is_punct(word):
    return unicodedata.category(word[0]) == 'Po'


def is_classifiable_word( word ):
    return word not in special_word2pos and not is_punct(word)

# ----------------------------------------------------------------------

corpus_path = sys.argv[1]
model_folder = sys.argv[-1]

# ----------------------------------------------------------------------

special_word2pos = dict()

with codecs.open( os.path.join(model_folder,'special_words.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        word = px[0]
        pos = px[1]
        special_word2pos[word] = pos

# ----------------------------------------------------------------------

# Загрузим тренировочный датасет.
print( 'Loading "{0}"...'.format(corpus_path) )
corpus = []
ntoken = 0
n_bad = 0
sent_count=0

word2freq = collections.Counter()
pos2index = dict()
tag_names = set()
tag2values = dict()
n_patterns = 0
all_chars = set()
max_word_len = 0

with codecs.open( corpus_path, 'r', 'utf-8') as rdr:
    sent = []
    good = True
    for line0 in rdr:
        line = line0.strip()
        if len(line)==0:
            if good and len(sent)>0:
                corpus.append( sent )
                ntoken += len( sent )
                #if len(corpus)>=max_sent:
                #    break
            else:
                n_bad += 1
            good = True
            sent = []
        else:
            tx = line.split('\t')
            if len(tx)<2:
                good = False
            else:
                word = normalize_word(tx[1])
                word2freq[word] += 1
                pos = tx[3] # часть речи

                if not is_classifiable_word(word) or pos not in salient_pos:
                    pos = u'???'
                    tags = []
                else:
                    features = tx[4].strip() if len(tx)==5 else u'' # теги
                    tags = [] if features==u'_' else features.split(u'|')

                tag_pairs = []

                max_word_len = max(max_word_len, len(word))
                all_chars.update( word_chars(word,INVERT) )

                if is_classifiable_word(word):
                    n_patterns += 1

                    if pos not in pos2index:
                        pos2index[pos] = len(pos2index)

                    for tag in tags:
                        tx = tag.split(u'=')
                        tag_name = tx[0]
                        tag_value = tx[1]

                        tag_names.add(tag_name)

                        if tag_name not in tag2values:
                            tag2values[tag_name] = set()

                        tag2values[tag_name].add(tag_value)
                        tag_pairs.append( (tag_name,tag_value) )

                sent.append( (word,pos,tag_pairs) )

print( 'done, {0} good sentences'.format(len(corpus)) )
print( 'max_word_len={}'.format(max_word_len) )
print( 'parts_of_speech={}'.format( unicode.join(u' ',pos2index) ) )
print( 'tag_names={}'.format( unicode.join( u' ', tag_names ) ) )

char2index = dict([ (c,i) for (i,c) in enumerate(all_chars) ])
bits_per_char = len(all_chars)

if BIDIR:
    bits_per_word = max_word_len * bits_per_char * 2
else:
    bits_per_word = max_word_len*bits_per_char
print( 'bits_per_char={0}'.format(bits_per_char) )

n_patterns = min( max_patterns, n_patterns )
print( 'n_patterns={0}'.format(n_patterns) )

# -----------------------------------------------------------------------------

# обучение классификатора частей речи

output_size_pos = len(pos2index)
print( 'output_size_pos={0}'.format(output_size_pos) )

with codecs.open( os.path.join(model_folder,'cn_char2index.dat'), 'w', 'utf-8' ) as wrt:
    for ch,id in char2index.iteritems():
        wrt.write( u'{0}\t{1}\n'.format(ch,id) )

with codecs.open( os.path.join(model_folder,'cn_pos2index.dat'), 'w', 'utf-8') as wrt:
    for pos,id in pos2index.iteritems():
        wrt.write( u'{0}\t{1}\n'.format(pos,id) )

with codecs.open( os.path.join(model_folder,'cn_tag_names.dat'), 'w', 'utf-8') as wrt:
    wrt.write( u'{}\n'.format( unicode.join( u' ', tag_names ) ) )

# ----------------------------------------------------------------------

print( 'Vectorization...' )

winspan = int((WINDOW-1)/2)

X_data = np.zeros( (n_patterns,WINDOW,bits_per_word), dtype=np.bool )
y_data = np.zeros( (n_patterns,output_size_pos), dtype=np.bool )
idata = 0
for sent in corpus:
    if idata==n_patterns:
        break
    nword = len(sent)
    for iword in range(nword):
        word0 = sent[iword][0]
        if is_classifiable_word(word0):
            pos = sent[iword][1]
            for j in range(WINDOW):
                word_index = iword-winspan+j
                word = u''
                if word_index>=0 and word_index<nword:
                    word = sent[word_index][0]

                if BIDIR:
                    for ichar, c in enumerate(word_chars(word, False)):
                        bit_index = ichar * bits_per_char + char2index[c]
                        X_data[idata, j, bit_index] = True
                    for ichar, c in enumerate(word_chars(word, True)):
                        bit_index = bits_per_char*max_word_len + ichar * bits_per_char + char2index[c]
                        X_data[idata, j, bit_index] = True
                else:
                    for ichar,c in enumerate(word_chars(word,INVERT)):
                        bit_index = ichar*bits_per_char + char2index[c]
                        X_data[ idata, j, bit_index ] = True

            y_data[ idata, pos2index[pos] ] = True
            idata += 1
            if idata==n_patterns:
                break

# ----------------------------------------------------------------------

model = Sequential()
HIDDEN_SIZE = 64
model.add( recurrent.LSTM( HIDDEN_SIZE, input_shape=(WINDOW, bits_per_word ) ) )
model.add( Dense( HIDDEN_SIZE, activation='relu' ) )
model.add( Dropout(0.1) )
model.add( Dense( output_size_pos, activation='softmax' ) )

#opt = keras.optimizers.SGD( lr=0.1, momentum=0.1, decay=0, nesterov=False )
opt = keras.optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#opt = keras.optimizers.Adagrad(lr=0.05, epsilon=1e-08)
#opt='rmsprop'

model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] )

open( os.path.join(model_folder,'cn_chars_postagger_net.arch'), 'w' ).write( model.to_json() )

model_checkpoint = ModelCheckpoint( os.path.join(model_folder,'cn_chars_postagger_net.model'), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping( monitor='val_acc', patience=5, verbose=1, mode='auto')

history = model.fit( X_data, y_data, validation_split=0.1, batch_size=64, nb_epoch=100, callbacks=[model_checkpoint,early_stopping] )

# ----------------------------------------------------------------------

tag2value2id = dict()
tag2output_size = dict()

# Теперь строим и обучаем отдельные модели классификации для каждого тега

for tag_name in tag_names:

    print( u'Building model for {}({})...'.format( tag_name, unicode.join(u' ',tag2values[tag_name]) ) )

    output_size = len( tag2values[tag_name] )+1 # добавляем 1 вариант 'тег отсутствует'
    tag2output_size[tag_name] = output_size
    value2id = dict( [ (v,i+1) for i,v in enumerate(tag2values[tag_name]) ] )
    tag2value2id[tag_name] = value2id

    y_data = np.zeros( (n_patterns,output_size), dtype=np.bool )
    idata = 0
    for sent in corpus:
        if idata==n_patterns:
            break
        nword = len(sent)
        for iword in range(nword):
            word0 = sent[iword][0]
            if is_classifiable_word(word0):
                tags = sent[iword][2]
                tag_value = 0
                for tag in tags:
                    if tag[0] == tag_name:
                        tag_value = value2id[tag[1]]
                        break

                y_data[idata, tag_value ] = True
                idata += 1
                if idata==n_patterns:
                    break

    model = Sequential()
    model.add(recurrent.LSTM(HIDDEN_SIZE, input_shape=(WINDOW, bits_per_word)))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    open( os.path.join(model_folder,u'chars_postagger_net_{0}.arch'.format(tag_name)), 'w').write(model.to_json())

    model_checkpoint = ModelCheckpoint( os.path.join(model_folder,u'chars_postagger_net_{0}.model'.format(tag_name)), monitor='val_acc', verbose=1,
                                       save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')

    history = model.fit(X_data, y_data, validation_split=0.1, batch_size=64, nb_epoch=100,
                        callbacks=[model_checkpoint, early_stopping])

# ----------------------------------------------------------------------

with codecs.open( os.path.join(model_folder,'cn_tag2output_size.dat'), 'w', 'utf-8' ) as wrt:
    for tag,output_size in tag2output_size.iteritems():
        wrt.write( u'{}\t{}\n'.format(tag,output_size) )

with codecs.open( os.path.join(model_folder,'cn_tag2value2id.dat'),'w','utf-8') as wrt:
    for tag,value2id in tag2value2id.iteritems():
        for value,id in value2id.iteritems():
            wrt.write( u'{}\t{}\t{}\n'.format(tag,value,id) )

# сохраняем общую конфигурацию
with codecs.open( os.path.join(model_folder,'cn_chars_postagger_net.config'),'w','utf-8') as cfg:
    params = {
              'WINDOW':WINDOW,
              'bits_per_char':bits_per_char,
              'bits_per_word':bits_per_word,
              'output_size_pos':output_size_pos,
              'max_word_len': max_word_len,
              'invert':INVERT,
              'bidir': BIDIR
             }
    json.dump( params, cfg )
