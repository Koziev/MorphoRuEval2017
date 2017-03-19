# -*- coding: utf-8 -*- 

'''
Эксперимент: тренировка различных POS Tagger алгоритмов из NLTK и сравнение их точности.
(c) Илья Козиев 2017 для morphrueval_17 inkoziev@gmail.com

Обучение:
python train_NLTK.py входной_корпус
'''

from __future__ import print_function
from nltk.tag import DefaultTagger
from nltk.tag import UnigramTagger
from nltk.tag import AffixTagger
from nltk.tag import tnt
from nltk.tag.sequential import ClassifierBasedPOSTagger
import codecs
import numpy as np
import pickle
import sys
import os.path

max_sent = 1000000

corpus_path = sys.argv[1]
model_folder = sys.argv[-1]

# ----------------------------------------------------------------------

classifiable_posx = { u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET' }
def classifiable_pos(part_of_speech):
    return True if part_of_speech in classifiable_posx else False

def normalize_word(word):
    return word.lower().replace(u'ё',u'е')

# ----------------------------------------------------------------------
special_word2pos = dict()

with codecs.open( 'special_words.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        word = px[0]
        pos = px[1]
        special_word2pos[word] = pos

# ----------------------------------------------------------------------

# Загрузим тренировочный датасет. 
print( 'Analyzing dataset {0}...'.format(corpus_path) )
corpus = []
ntoken = 0
n_bad = 0
tagset2id = dict()
with codecs.open( corpus_path, 'r', 'utf-8') as rdr:
    sent = []
    good = True
    for line0 in rdr:
        line = line0.strip()
        if len(line)==0:
            if good and len(sent)>0:
                corpus.append( sent )
                ntoken += len( sent )
            else:
                n_bad += 1
            good = True
            sent = []
        else:
            tx = line.split(u'\t')
            if len(tx)<4:
                good = False
            else:
                word = normalize_word(tx[1]) # слово
                pos = tx[3] # часть речь
                tagset = u'???'

                if word in special_word2pos:
                    tagset = special_word2pos[word]
                elif classifiable_pos(pos):
                    features = tx[4] if len(tx)==5 else u'' # теги
                    tagset = (pos + u' ' + features).strip()
                else:
                    tagset = u'???'

                if tagset not in tagset2id:
                    tagset2id[tagset] = len(tagset2id)

                t_id = tagset2id[tagset]
                sent.append( (word,t_id) )

print( 'done, {0} good sentences, {1} tokens'.format(len(corpus),ntoken) )
print( 'tagset2id.count={0}'.format(len(tagset2id)))

# ----------------------------------------------------------------------

with codecs.open( os.path.join(model_folder,'nltk_tagset2id.dat'),'w','utf-8') as wrt:
    for tagset,t_id in tagset2id.iteritems():
        wrt.write( u'{0}\t{1}\n'.format(tagset,t_id))

# ----------------------------------------------------------------------

n_patterns = len(corpus)

n_test = int(n_patterns*0.1)
n_train = n_patterns-n_test
print( 'n_test={0} n_train={1}'.format(n_test,n_train) )
data_indeces = [ x for x in range(n_patterns) ]
np.random.shuffle( data_indeces )
test_indeces = data_indeces[ : n_test ]
train_indeces = data_indeces[ n_test : ]

train_corpus = [ corpus[i] for i in train_indeces ]
test_corpus = [ corpus[i] for i in test_indeces ]


# ----------------------------------------------------------------------

#default_tagger = DefaultTagger(u'NOUN')

# # ----------------------------------------------------------------------
#
# print( 'Training AffixTagger on 1-suffixes...' )
# suffix1_tagger = AffixTagger( train_corpus, affix_length=-1, backoff=default_tagger )
# print( 'Testing...' )
# acc = suffix1_tagger.evaluate(test_corpus)
# print( 'AffixTagger(1) accuracy={0}\n'.format(acc) )
#
# # ----------------------------------------------------------------------
#
# print( 'Training AffixTagger on 2-suffixes...' )
# suffix2_tagger = AffixTagger( train_corpus, affix_length=-2, backoff=suffix1_tagger )
# print( 'Testing...' )
# acc = suffix2_tagger.evaluate(test_corpus)
# print( 'AffixTagger(2,1) accuracy={0}\n'.format(acc) )
#
# # ----------------------------------------------------------------------
#
# print( 'Training AffixTagger on 3-suffixes...' )
# suffix3_tagger = AffixTagger( train_corpus, affix_length=-3, backoff=suffix2_tagger )
# print( 'Testing...' )
# acc = suffix3_tagger.evaluate(test_corpus)
# print( 'AffixTagger(3,2,1) accuracy={0}\n'.format(acc) )
#
# # ----------------------------------------------------------------------
#
# print( 'Training AffixTagger on 4,3,2-suffixes...' )
# suffix4_tagger = AffixTagger( train_corpus, affix_length=-4, backoff=suffix3_tagger )
# print( 'Testing...' )
# acc = suffix4_tagger.evaluate(test_corpus)
# print( 'AffixTagger(4,3,2) accuracy={0}\n'.format(acc) )
#
# # ----------------------------------------------------------------------
#
# print( 'Testing UnigramTagger + AffixTagger(4,3,2,1)...' )
# unigram_tagger = UnigramTagger(train_corpus, backoff=suffix4_tagger )
# acc = unigram_tagger.evaluate(test_corpus)
# print( 'UnigramTagger+AffixTagger(4,3,2,1) accuracy={0}\n'.format(acc) )
#
# # ----------------------------------------------------------------------
#
# print( 'Training TnT...' )
# tnt_tagger = tnt.TnT()
# tnt_tagger.train(train_corpus)
# print( 'Testing...' )
# acc = tnt_tagger.evaluate(test_corpus)
# print( 'TnT accuracy={0}\n'.format(acc) )
#
# # ----------------------------------------------------------------------
#
# print( 'Training UnigramTagger...' )
# unigram_tagger = UnigramTagger(train_corpus)
# with open( 'unigram.pos_tagger.pickle', 'wb' ) as f:
#     pickle.dump( unigram_tagger, f )
#
# print( 'Testing...' )
# acc = unigram_tagger.evaluate(test_corpus)
# print( 'UnigramTagger accuracy={0}\n'.format(acc) )


# ----------------------------------------------------------------------

print( 'Training ClassifierBasedPOSTagger...' )
cbt = ClassifierBasedPOSTagger( train=train_corpus)
print( 'Testing...' )
acc = cbt.evaluate(test_corpus)
print( 'accuracy={0}\n'.format(acc) )

print( 'Storing...' )
with open( os.path.join(model_folder,'ClassifierBasedPOSTagger.pickle'), 'wb') as f:
    pickle.dump(cbt,f)
