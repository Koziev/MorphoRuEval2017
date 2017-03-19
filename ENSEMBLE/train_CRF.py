# -*- coding: utf-8 -*-
'''
Из POSTagger-корпуса создаем обучающий датасет для CRFSuite.
'''

from __future__ import print_function
import codecs
import collections

import sys

max_sent = 1000000

src_dataset = sys.argv[1]

crftrain_dataset = r'crfsuite_train.dat'
crftest_dataset = r'crfsuite_test.dat'

USE_SUFFIX_FEATURE = True
MIN_SUFFIX_LEN = 3
MAX_SUFFIX_LEN = 3
WINDOW=3 # полный размер окна контекста

MAX_PATTERNS = 1000000

HOLDOUT_Nth = 10 # каждый N-ый сэмпл (предложение/паттерн) в тестовый набор

USE_LEXICON = False # добавлять морфологические теги слов из word2tag.dat

ADD_WORDCLUSTER_FEATURES = True # добавлять номера тегов слов из word2cluster (получены кластеризацией w2v-векторов слов)

ADD_WORD_FEATURES = True # добавлять самые частотные слова как самостоятельные фичи
MIN_FEATUREWORD_FREQ = 100

# ------------------------------------------------------------        

win = (WINDOW-1)/2

# ------------------------------------------------------------        

word2tags = dict()
word2cluster = dict()
featured_words = set()


def extract_features( word, feature_name_prefix ):
    features = []
    
    if word==u'':
        features.append( feature_name_prefix+u'<nil>' )
    else:    
        lword = word.lower()    
        if word[0].isupper() and word[1:].islower():
            features.append( feature_name_prefix+u'Aa' )
        elif word.isupper():
            features.append( feature_name_prefix+u'AA' )

        wlen = len(word)
        if USE_SUFFIX_FEATURE:
            if wlen>2:
                for suffix_len in range(MIN_SUFFIX_LEN,min(MAX_SUFFIX_LEN,wlen-1)+1):
                    suffix = lword[ wlen-suffix_len : wlen ]
                    features.append( feature_name_prefix + u'sfx='+suffix )
                    #print( u'DEBUG word={0} suffix_len={1} suffix={2}'.format(word,suffix_len,suffix) )
                    #raw_input( 'press a key' )
            else:
                features.append( feature_name_prefix+u'word='+lword )
            
        if lword in word2tags:
            tags = word2tags[lword]
            for tag in tags:
                features.append( feature_name_prefix+u'tag='+tag )
                
        if lword in word2cluster:
            features.append( feature_name_prefix+u'cluster='+word2cluster[lword] )

        if lword in featured_words:
            features.append( feature_name_prefix+u'featured_word='+lword )
            
    return features

# ------------------------------------------------------------        

if USE_LEXICON==True:
    with codecs.open( 'word2tags.dat', 'r', 'utf-8-sig' ) as rdr:
        print( 'Loading lexicon...' )
        for line in rdr:
            parts = line.strip().split(u'\t')
            word = parts[0].lower()
            part_of_speech = parts[1]
            tags = set( parts[2].split(u' ') ) if len(parts)==3 else set()
            tags.add( part_of_speech )
            
            if word not in word2tags:
                word2tags[word] = tags
            else:
                word2tags[word].update(tags)

    print( '{0} words in lexicon'.format(len(word2tags)) )

# ------------------------------------------------------------        

if ADD_WORDCLUSTER_FEATURES==True:
    with codecs.open( 'word2cluster.dat','r','utf-8') as rdr:
        print( 'Loading word2cluster...' )
        for line in rdr:
            parts = line.strip().split(u'\t')
            word = parts[0].lower()
            word2cluster[word] = parts[1]

# ------------------------------------------------------------        
            
if ADD_WORD_FEATURES==True:
    word2freq = collections.Counter()
    with codecs.open( src_dataset, 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            if len(parts)>=4:
                word = parts[1].lower()
                word2freq[word] += 1
    
    featured_words = set( [ word for word,cnt in filter( lambda z:z[1]>=MIN_FEATUREWORD_FREQ, word2freq.iteritems() ) ] )
    
    print( 'There are {0} featured words'.format( len(featured_words) ) )

# ------------------------------------------------------------        

print( 'Building tagsets...' )
tagset2id = dict()
total_nb_sent = 0
with codecs.open( src_dataset, 'r', 'utf-8' ) as rdr:
    for line in rdr:
        parts = line.strip().split(u'\t')
        if len(parts)>=4:
            tagset = parts[3] # part of speech
            if len(parts)==5:
                tagset = tagset + u' ' + parts[4] # append tags
            if tagset not in tagset2id:
                tagset2id[tagset] = len(tagset2id)
        else:
            # пустая строка - разделитель предложений
            total_nb_sent += 1

print( 'total number of sentences={0}'.format(total_nb_sent) )            
print( 'number of tagsets={0}'.format(len(tagset2id)) )
            
with codecs.open( 'id2tagset.dat', 'w', 'utf-8' ) as wrt:
    for tagset,id in tagset2id.iteritems():
        wrt.write( u'{0}\t{1}\n'.format(id,tagset) )

# ------------------------------------------------------------        

print( 'Converting...' )

sent_count=0
pattern_count=0
rdr = codecs.open( src_dataset, 'r', 'utf-8' )
wrt1_train = codecs.open( crftrain_dataset, 'w', 'utf-8' )
wrt1_test = codecs.open( crftest_dataset, 'w', 'utf-8' )

sent = []
for line in rdr:

    if pattern_count>=MAX_PATTERNS:
        break

    parts = line.strip().split(u'\t')
    if len(parts)==0 or len(parts[0])==0:
        # end of sentence
        sent_count += 1
        if (sent_count%1000)==0:
            print( '{0}/{1}'.format(sent_count,total_nb_sent), end='\r' )

        nword = len(sent)
        for i in range(nword):
            token_features = []
            for j in range(WINDOW):
                word_index = i-win+j
                word = u''
                if word_index>=0 and word_index<nword:
                    word = sent[word_index][0]
                token_features += extract_features( word, u'['+str(-win+j)+u']' )
            
            target_tagset = tagset2id[sent[i][1]]
            tags = u'\t'.join( token_features )
            #print( u'{0} --> {1}'.format( sent[i][0], tags ) )
            #raw_input('press a key...' )
            
            if (sent_count%HOLDOUT_Nth)==0:
                wrt1_test.write( u'{0}\t{1}\n'.format(target_tagset, tags ) )
            else:
                wrt1_train.write( u'{0}\t{1}\n'.format(target_tagset, tags ) )

            pattern_count += 1
            if pattern_count>=MAX_PATTERNS:
                break

        if (sent_count%HOLDOUT_Nth)==0:
            wrt1_test.write('\n')
        else:
            wrt1_train.write( '\n' )

        sent = []
    else:
        word = parts[1]
        tagset = parts[3] # part of speech
        if len(parts)==5:
            tagset = tagset + u' ' + parts[4] # tags
        sent.append( (word,tagset) )

wrt1_train.close()
wrt1_test.close()

print( 'Generation complete.' )
