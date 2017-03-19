# -*- coding: utf-8 -*- 
'''
POS Tagger на базе ансамбля моделей
(c) Илья Козиев 2017 для morphorueval 2017 inkoziev@gmail.com

Разметка входного корпуса с использованием ранее обученных моделей:
python apply_ensemble.py tag входной_корпус выходной_файл

Оценка модели по размеченному корпусу:
python apply_ensemble.py eval эталонный_корпус

'''

from __future__ import print_function

import codecs
import collections
import json
import sys
import numpy as np
import pickle
import scipy.sparse
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import unicodedata
import unicodedata
import os.path

USE_NLTK = True
USE_DECISIONTREE = True
USE_CHARNET = True
USE_MAXENT = True
USE_MODEL1 = True

if USE_CHARNET or USE_MODEL1:
    from keras.models import Sequential
    from keras.layers.core import Activation
    from keras.layers.core import Dense, Dropout
    from keras.models import model_from_json




run_mode = '' # tag или eval

# ----------------------------------------------------------------------

DT_MIN_SUFFIX_LEN = 0
DT_MAX_SUFFIX_LEN = 0
DT_ADD_WORDCLUSTER_FEATURES = False
DT_BORDER_WORD = u''

# функция возвращает список признаков для слова для моделей DecisionTreeClassifier
# и LogisticRegression.
def dt_extract_features(word, feature_name_prefix):
    features = []

    if word == DT_BORDER_WORD:
        features.append(feature_name_prefix + u'<nil>')
    else:
        lword = word.lower()
        if word[0].isupper() and word[1:].islower():
            features.append(feature_name_prefix + u'Aa')
        elif word.isupper():
            features.append(feature_name_prefix + u'AA')

        wlen = len(word)
        if wlen > 2:
            for suffix_len in range(DT_MIN_SUFFIX_LEN, min(DT_MAX_SUFFIX_LEN, wlen - 1) + 1):
                suffix = lword[wlen - suffix_len: wlen]
                features.append(feature_name_prefix + u'sfx=' + suffix)
                # print( u'DEBUG word={0} suffix_len={1} suffix={2}'.format(word,suffix_len,suffix) )
                # raw_input( 'press a key' )
        else:
            features.append(feature_name_prefix + u'word=' + lword)

        if lword in dt_word2cluster:
            features.append(feature_name_prefix + u'cluster=' + dt_word2cluster[lword])

        if lword in dt_featured_words:
            features.append(feature_name_prefix + u'featured_word=' + lword)

    return features

    
def normalize_word(word):
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


def limit_to( l, maxlen ):
    if len(l)>maxlen:
        return l[:maxlen]
    else:
        return l

classifiable_posx = {u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET'}
def classifiable_pos(part_of_speech):
    return True if part_of_speech in classifiable_posx else False


# ----------------------------------------------------------------------

def usage():
    print( 'Usage:' )
    print( 'python apply_ensemble.py tag  input_corpus output_file  model_datafiles_folder' )
    print( 'python apply_ensemble.py eval input_corpus results_file model_datafiles_folder' )
    exit()

if len(sys.argv)!=5:
    usage()

run_mode = sys.argv[1]

if run_mode!='tag' and run_mode!='eval':
    print( 'Unknown scenario {0}'.format(run_mode) )
    usage()

is_tagging=True
if run_mode=='eval':
    is_tagging=False

input_path = sys.argv[2]
result_path = sys.argv[3]
model_folder = sys.argv[4]

if is_tagging:
    print( u'Tagging data in {0} and writing results to {1}...'.format(input_path,result_path) )
else:
    print( u'Model evaluation on {0}...'.format(input_path) )


# ----------------------------------------------------------------------

print( 'Loading models...' )


# ----------------------------------------------------------------------
special_word2pos = dict()
DET_forms = set()

with codecs.open( os.path.join(model_folder,'DET_forms.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        word = line.strip()
        DET_forms.add(word)

with codecs.open( os.path.join(model_folder,'special_words.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        word = px[0]
        pos = px[1]
        special_word2pos[word] = pos

def is_punct(word):
    return unicodedata.category(word[0]) == 'Po'

pos2tags = dict()
with codecs.open( os.path.join(model_folder,'pos2tags.dat'), 'r', 'utf-8' ) as rdr:
    for line in rdr:
        parts = line.split( u'\t' )
        if len(parts)==2:
            part_of_speech = parts[0]
            tags = set( parts[1].strip().split(u' ') )
            pos2tags[part_of_speech] = tags


word2freq = dict()
with codecs.open( os.path.join(model_folder,'word2freq.dat'), 'r', 'utf-8' ) as rdr:
    for line in rdr:
        parts = line.strip().split( u'\t' )
        word = parts[0]
        freq = int(parts[1])
        word2freq[word] = freq

word2tagset = dict()
with codecs.open( os.path.join(model_folder,'word2tagset.dat'), 'r', 'utf-8' ) as rdr:
    for line in rdr:
        parts = line.strip().split( u'\t' )
        if len(parts)==2:
            px = parts[1].split(u' ')
            part_of_speech = px[0]
            tags = []
            for tag in px[1].split(u'|'):
                tx = tag.split(u'=')
                if len(tx)==2:
                    tags.append( (tx[0],tx[1]) )
            word2tagset[ parts[0] ] = (part_of_speech,tags)

# -------------------------------------------------------------------------

nltk_id2tagset = dict()
with codecs.open( os.path.join(model_folder,'nltk_tagset2id.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        parts = line.split(u'\t')
        if len(parts)==2:
            tagset = parts[0]
            t_id = int(parts[1].strip())

            tx = tagset.split(u' ')
            part_of_speech = tx[0]
            tags = []

            if len(tx)==2:
                for tag in tx[1].split(u'|'):
                    if tag!=u'_':
                        tag_name,tag_value = tag.split(u'=')
                        tags.append( (tag_name,tag_value) )
            nltk_id2tagset[t_id] = (part_of_speech,tags)

if USE_NLTK:
    with open(os.path.join(model_folder,'ClassifierBasedPOSTagger.pickle'), 'rb') as f:
        nltk_cl = pickle.load(f)

# ------------------------------------------------------------------------

# TODO весь код для DecisionTreeClassifier и MaxEnt вынести в классы, там инкапсулировать
# конфигурацию и т.д.
DT_WINDOW = 0
DT_ADD_WORDCLUSTER_FEATURES = False
dt_nb_features = -1
dt_nb_labels = -1

with open( os.path.join(model_folder,'dt_postagger.config'), 'rt') as cfg:
    params = json.load(cfg)
    DT_WINDOW = int(params['WINDOW'])
    DT_MIN_SUFFIX_LEN = int(params['MIN_SUFFIX_LEN'])
    DT_MAX_SUFFIX_LEN = int(params['MAX_SUFFIX_LEN'])
    DT_ADD_WORDCLUSTER_FEATURES = bool(params['ADD_WORDCLUSTER_FEATURES'])
    DT_BORDER_WORD = params['BORDER_WORD']
    dt_nb_features = int(params['nb_features'])
    dt_nb_labels = int(params['nb_labels'])

dt_word2cluster = dict() # 'word2cluster.dat'
dt_featured_words = set() # 'dt_featured_words.dat'
dt_id2pos = dict() # 'dt_pos2id.dat'
dt_salient_tags = set() # 'dt_salient_tags.dat'
dt_feature2id = dict() # 'dt_feature2id.dat'
dt_tag2id = dict()
dt_tag2id2value = dict()

dt_tag2decisiontree = dict()
dt_tag2maxent = dict()

with codecs.open( os.path.join(model_folder,'dt_codebook.dat'),'r','utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        tag_name = tx[0]
        tag_id = int(tx[1])
        dt_tag2id[tag_name] = tag_id
        dt_tag2id2value[tag_name] = dict()
        for v in tx[2].split(u' '):
            vx = v.split(u':')
            value_id = int(vx[0])
            value_name = vx[1]
            dt_tag2id2value[tag_name][value_id] = value_name

        #dt_tag2decisiontree = dict()
        #dt_tag2maxent = dict()
        if USE_DECISIONTREE:
            with open( os.path.join(model_folder,'DecisionTreeClassifier_{}.pickle'.format(tag_id)), 'rb' ) as f:
                m = pickle.load(f)
                dt_tag2decisiontree[tag_name] = m

        if USE_MAXENT:
            with open( os.path.join(model_folder,'LogisticRegression_{}.pickle'.format(tag_id)), 'rb' ) as f:
                m = pickle.load(f)
                dt_tag2maxent[tag_name] = m


if DT_ADD_WORDCLUSTER_FEATURES==True:
    with codecs.open( os.path.join(model_folder,'word2cluster.dat'),'r','utf-8') as rdr:
        print( 'Loading word2cluster...' )
        for line in rdr:
            parts = line.strip().split(u'\t')
            word = parts[0].lower()
            dt_word2cluster[word] = parts[1]

with codecs.open( os.path.join(model_folder,'dt_featured_words.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        dt_featured_words.add( line.strip() )

with codecs.open(os.path.join(model_folder,'dt_feature2id.dat'),'r','utf-8') as rdr:
    for line in rdr:
        parts = line.strip().split(u'\t')
        feature = parts[0]
        f_id = int(parts[1])
        dt_feature2id[feature] = f_id

with codecs.open( os.path.join(model_folder,'dt_pos2id.dat'), 'r', 'utf-8' ) as rdr:
    for line in rdr:
        parts = line.strip().split(u'\t')
        part_of_speech = parts[0]
        f_id = int(parts[1])
        dt_id2pos[f_id] = part_of_speech

dt_tag2id = dict()
dt_tag2values = dict()
with codecs.open( os.path.join(model_folder,'dt_salient_tags.dat'), 'r', 'utf-8' ) as rdr:
    for line in rdr:
        parts = line.strip().split(u'\t')
        t_id = int(parts[0])
        tag_name = parts[1]
        tag_values = parts[2].strip().split(u' ')
        dt_tag2id[tag_name] = id
        dt_tag2values[tag_name] = tag_values

if USE_DECISIONTREE:
    with open( os.path.join(model_folder,'DecisionTreeClassifier.pickle'), 'rb' ) as f:
        dt_cl = pickle.load(f)

if USE_MAXENT:
    with open( os.path.join(model_folder,'LogisticRegression.pickle'), 'rb' ) as f:
        dt_maxent = pickle.load(f)

# ------------------------------------------------------------------------
# todo - перенести код и структуры данных для char_postagger_net в класс.
cn_char2index = dict() # 'cn_char2index.dat'
cn_id2pos = dict() # 'cn_pos2index.dat'
cn_invert = False
cn_bidir = False
cn_tag_names = set()
cn_tag2model = dict() # сеточный классификатор для каждого тега
cn_tag2output_size = dict() # число значений на выходе классификатора для каждого тега
cn_tag2id2value = dict() # для декодирования результатов работы классификаторов для каждого тега

with codecs.open( os.path.join(model_folder,'cn_char2index.dat'),'r', 'utf-8') as rdr:
    for line in rdr:
        px = line.split(u'\t')
        ch = px[0]
        ch_id = int(px[1].strip())
        cn_char2index[ch] = ch_id

with codecs.open( os.path.join(model_folder,'cn_pos2index.dat'),'r', 'utf-8') as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        part_of_speech = px[0]
        id = int(px[1])
        cn_id2pos[id] = part_of_speech

CN_WINDOW = -1
cn_bits_per_char = -1
cn_bits_per_word = -1
cn_output_size_pos = -1

with open(os.path.join(model_folder,'cn_chars_postagger_net.config'), 'rt') as cfg:
    params = json.load(cfg)
    CN_WINDOW = int(params['WINDOW'])
    cn_bits_per_char = int(params['bits_per_char'])
    cn_bits_per_word = int(params['bits_per_word'])
    cn_output_size_pos = int(params['output_size_pos'])
    cn_max_word_len = int(params['max_word_len'])
    cn_invert = bool(params['invert'])
    cn_bidir = bool(params['bidir'])

with open(os.path.join(model_folder,'cn_tag_names.dat'), 'rt') as rdr:
    cn_tag_names = rdr.readline().strip().split(u' ')

    
with codecs.open( os.path.join(model_folder,'cn_tag2output_size.dat'), 'r', 'utf-8' ) as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        tag = px[0]
        output_size = int(px[1])
        cn_tag2output_size[tag] = output_size

with codecs.open( os.path.join(model_folder,'cn_tag2value2id.dat'),'r','utf-8') as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        tag = px[0]
        value = px[1]
        v_id = int(px[2])
        if tag not in cn_tag2id2value:
            cn_tag2id2value[tag] = dict()
        cn_tag2id2value[tag][v_id] = value

    
if USE_CHARNET:
    cn_model = model_from_json(open(os.path.join(model_folder,'cn_chars_postagger_net.arch')).read())
    cn_model.load_weights(os.path.join(model_folder,'cn_chars_postagger_net.model'))
    
    for tag_name in cn_tag_names:
        model_filename = u'chars_postagger_net_{0}'.format(tag_name)
        
        cn_tag_model = model_from_json(open(os.path.join(model_folder,model_filename+u'.arch')).read())
        cn_tag_model.load_weights(os.path.join(model_folder,model_filename+u'.model'))
        cn_tag2model[tag_name] = cn_tag_model


# ------------------------------------------------------------------------

# данные для MODEL1

m1_invert = False
m1_bidir = False
m1_max_word_len = -1
m1_id2tag = dict()
m1_char2index = dict()

with codecs.open(os.path.join(model_folder,'m1_char2index.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        parts = line.split(u'\t')
        if len(parts) == 2:
            m1_char2index[parts[0]] = int(parts[1].strip())

with codecs.open(os.path.join(model_folder,'m1_id2tag.dat'), 'r', 'utf-8') as rdr:
    for line in rdr:
        parts = line.split(u'\t')
        if len(parts) == 2:
            t_id = int(parts[0])
            tagset = parts[1]
            tx = tagset.split(u' ')
            part_of_speech = tx[0].strip()

            tags = []
            if len(tx) == 2:
                for tag in tx[1].strip().split(u'|'):
                    if tag != u'_':
                        tag_name, tag_value = tag.split(u'=')
                        tags.append((tag_name, tag_value))

            m1_id2tag[t_id] = (part_of_speech,tags)

with open(os.path.join(model_folder,'m1_chars_postagger_net.config'), 'rt') as cfg:
    params = json.load(cfg)
    M1_WINDOW = int(params['WINDOW'])
    m1_bits_per_char = int(params['bits_per_char'])
    m1_bits_per_word = int(params['bits_per_word'])
    m1_output_size = int(params['output_size'])
    m1_invert = bool(params['invert'])
    m1_bidir = bool(params['bidir'])
    m1_max_word_len = int(params['max_word_len'])

if USE_MODEL1:
    model1 = model_from_json(open(os.path.join(model_folder,'m1_chars_postagger_net.arch')).read())
    model1.load_weights(os.path.join(model_folder,'m1_chars_postagger_net.model'))

# ------------------------------------------------------------------------

print( 'Start processing input corpus {0}...'.format(input_path) )

rdr_corpus = codecs.open( input_path, 'r', 'utf-8' )

wrt_result = codecs.open( result_path, 'w', 'utf-8' )

line_num = 0
processed_sent_count = 0
total_word_count = 0
incorrect_pos_count = 0
incorrect_word_count = 0
sent = []
good = True
for line0 in rdr_corpus:
    line = line0.strip()
    line_num += 1
    if len(line)==0:
        if good and len(sent)>0:

            all_known = True
            for wdata in sent:
                word = wdata[0]
                if word not in word2freq:
                    all_known = False
                    break

            sent_arg = [ wdata[0] for wdata in sent ]

            nltk_result = []
            if all_known and USE_NLTK:
                nltk_result = nltk_cl.tag( sent_arg )

            nword = len(sent)
            for iword in range(nword):

                word = sent[iword][0]
                raw_data = sent[iword][1]

                part_of_speech = u''
                tags = []

                if word in word2tagset:
                    tagset = word2tagset[word]
                    part_of_speech = tagset[0]
                    tags = tagset[1]
                elif is_punct(word):
                    part_of_speech = u'PUNCT'
                    tags = []
                elif word in special_word2pos:
                    part_of_speech = special_word2pos[word]
                    tags = []
                else:
                    # тут запускается ансамбль моделей для определения части речи и всех тегов
                    pos2score = collections.Counter()
                    tag2value2score = dict()

                    if USE_NLTK:
                        if len(nltk_result)>0:
                            nltk_tagset_id = nltk_result[iword][1]
                            nltk_tagset = nltk_id2tagset[nltk_tagset_id]
                            nltk_part_of_speech = nltk_tagset[0]
                            pos2score[nltk_part_of_speech] += 1

                            for tag in nltk_tagset[1]:
                                tag_name = tag[0]
                                tag_value = tag[1]
                                if tag_name not in tag2value2score:
                                    tag2value2score[tag_name] = collections.Counter()

                                tag2value2score[tag_name][tag_value] += 1

                    # DecisionTreeClassifier
                    features = []
                    for j in range(DT_WINDOW):
                        jword = iword - (DT_WINDOW - 1) / 2 + j
                        word2 = DT_BORDER_WORD
                        if jword >= 0 and jword < nword:
                            word2 = sent[jword][0]
                        prefix = u'[{}]'.format(-(DT_WINDOW - 1) / 2 + j)
                        features.extend(dt_extract_features(word2, prefix))

                    if USE_DECISIONTREE or USE_MAXENT:
                        dt_X_data = scipy.sparse.lil_matrix((1, dt_nb_features))
                        for feature in features:
                            if feature in dt_feature2id: # в тестовой выборке могут быть новые окончания, игнорируем их
                                f_id = dt_feature2id[feature]
                                dt_X_data[0,f_id] = True

                    if USE_DECISIONTREE:
                        dt_y = dt_cl.predict(dt_X_data)
                        dt_part_of_speech = dt_y[0]
                        pos2score[ dt_id2pos[dt_part_of_speech] ] += 1

                    if USE_MAXENT:
                        dt_y = dt_maxent.predict(dt_X_data)
                        dt_part_of_speech = dt_y[0]
                        pos2score[ dt_id2pos[dt_part_of_speech] ] += 1

                    if USE_CHARNET:
                        # создавать входной тензор для каждого слова - это конечно плохо,
                        # надо вынести в поле класса и просто очищать.
                        cn_X_data = np.zeros((1, CN_WINDOW, cn_bits_per_word), dtype=np.bool)

                        for j in range(CN_WINDOW):
                            word_index = iword - (CN_WINDOW-1)/2 + j
                            word2 = u''
                            if word_index >= 0 and word_index < nword:
                                word2 = sent[word_index][0]

                            if cn_bidir:
                                for ichar, c in enumerate(limit_to(word_chars(word2, False), cn_max_word_len)):
                                    if c in cn_char2index:
                                        bit_index = ichar * cn_bits_per_char + cn_char2index[c]
                                        cn_X_data[0, j, bit_index] = True

                                for ichar, c in enumerate(limit_to(word_chars(word2, True), cn_max_word_len)):
                                    if c in cn_char2index:
                                        bit_index = cn_max_word_len*cn_bits_per_char + ichar * cn_bits_per_char + cn_char2index[c]
                                        cn_X_data[0, j, bit_index] = True
                            else:
                                for ichar, c in enumerate(limit_to(word_chars(word2,cn_invert),cn_max_word_len)):
                                    if c in cn_char2index:
                                        bit_index = ichar * cn_bits_per_char + cn_char2index[c]
                                        cn_X_data[ 0, j, bit_index] = True

                        cn_y = cn_model.predict_classes(cn_X_data,verbose=0)
                        cn_part_of_speech = cn_y[0]
                        pos2score[ cn_id2pos[cn_part_of_speech] ] += 1

                    if USE_MODEL1:
                        # --------------- MODEL1 --------------------------
                        m1_X_data = np.zeros((1, M1_WINDOW, m1_bits_per_word), dtype=np.bool)

                        for j in range(M1_WINDOW):
                            word_index = iword-(M1_WINDOW-1)/2 + j
                            word2 = u''
                            if word_index>=0 and word_index<nword:
                                word2 = sent[word_index][0]

                            if m1_bidir:
                                for ichar, c in enumerate(limit_to(word_chars(word2, False),m1_max_word_len)):
                                    if c in cn_char2index:
                                        bit_index = ichar * m1_bits_per_char + m1_char2index[c]
                                        m1_X_data[0, j, bit_index] = True

                                for ichar, c in enumerate(limit_to(word_chars(word2, True),m1_max_word_len)):
                                    if c in cn_char2index:
                                        bit_index = m1_max_word_len*m1_bits_per_char + ichar * m1_bits_per_char + m1_char2index[c]
                                        m1_X_data[0, j, bit_index] = True
                            else:
                                for ichar,c in enumerate(limit_to(word_chars(word2,m1_invert),m1_max_word_len)):
                                    if c in cn_char2index:
                                        bit_index = ichar*m1_bits_per_char + m1_char2index[c]
                                        m1_X_data[ 0, j, bit_index ] = True

                        m1_y = model1.predict_classes(m1_X_data, verbose=0)
                        tagset_id = m1_y[0]
                        tagset = m1_id2tag[tagset_id]
                        m1_part_of_speech = tagset[0]
                        m1_tags = tagset[1]

                        pos2score[m1_part_of_speech] += 1

                        for tag in m1_tags:
                            tag_name = tag[0]
                            tag_value = tag[1]
                            if tag_name not in tag2value2score:
                                tag2value2score[tag_name] = collections.Counter()

                            tag2value2score[tag_name][tag_value] += 1


                    # todo ... остальные модели голосуют по части речи

                    # выбираем часть речи по итогам голосования
                    best_pos = u''
                    best_pos_score = 0
                    for pos,score in pos2score.iteritems():
                        if score>best_pos_score:
                            best_pos_score = score
                            best_pos = pos

                    part_of_speech = best_pos

                    # charnet-модель выбирает значение каждого тега для победившей части речи
                    if USE_CHARNET:
                        if part_of_speech in pos2tags:
                            for tag in pos2tags[part_of_speech]:
                                if  tag in cn_tag2model:
                                    tag_model = cn_tag2model[tag]
                                    cn_y = tag_model.predict_classes(cn_X_data,verbose=0)
                                    cn_value_id = cn_y[0]

                                    if cn_value_id==0:
                                        value_name = u'' # специальный случай - у тега нет значения (тега нет в тегсете)
                                    else:
                                        value_name = cn_tag2id2value[tag][cn_value_id]

                                    if tag not in tag2value2score:
                                        tag2value2score[tag] = collections.Counter()

                                    tag2value2score[tag][value_name] += 1
                                
                    if USE_DECISIONTREE:
                        if part_of_speech in pos2tags:
                            for tag in pos2tags[part_of_speech]:
                                if  tag in dt_tag2decisiontree:
                                    tag_model = dt_tag2decisiontree[tag]
                                    dt_y = tag_model.predict(dt_X_data)
                                    dt_value_id = dt_y[0]

                                    if dt_value_id==0:
                                        value_name = u'' # специальный случай - у тега нет значения (тега нет в тегсете)
                                    else:
                                        value_name = dt_tag2id2value[tag][dt_value_id]

                                    if tag not in tag2value2score:
                                        tag2value2score[tag] = collections.Counter()

                                    tag2value2score[tag][value_name] += 1


                    if USE_MAXENT:
                        if part_of_speech in pos2tags:
                            for tag in pos2tags[part_of_speech]:
                                if  tag in dt_tag2maxent:
                                    tag_model = dt_tag2maxent[tag]
                                    dt_y = tag_model.predict(dt_X_data)
                                    dt_value_id = dt_y[0]

                                    if dt_value_id==0:
                                        value_name = u'' # специальный случай - у тега нет значения (тега нет в тегсете)
                                    else:
                                        value_name = dt_tag2id2value[tag][dt_value_id]

                                    if tag not in tag2value2score:
                                        tag2value2score[tag] = collections.Counter()

                                    tag2value2score[tag][value_name] += 1

                    # выбор победителей по тегам
                    # учесть применимость тегов для победившей части речи
                    tags = []

                    if part_of_speech in pos2tags:
                        for tag in pos2tags[part_of_speech]:
                            if tag in tag2value2score:
                                best_value = u''
                                best_score = 0
                                for value,score in tag2value2score[tag].iteritems():
                                    if score>best_score:
                                        best_value = value
                                        best_score = score
                                if best_score>0 and best_value!=u'':
                                    tags.append( (tag,best_value) )

                tags_str = u'_' if len(tags)==0 else unicode.join( u'|', [ tag_name+u'='+tag_value for (tag_name,tag_value) in tags ] )

                if word in DET_forms and part_of_speech == u'ADJ':
                    part_of_speech = u'DET'

                #if part_of_speech == u'DET':
                #    tags_list = remove_tag([u'Degree', u'Animacy'], tags_list)

                total_word_count += 1
                if is_tagging:
                    wrt_result.write( u'{0}\t{1}\t{2}\t{3}\t{4}\n'.format(raw_data[0],raw_data[1],raw_data[2],part_of_speech,tags_str) )
                elif classifiable_pos(raw_data[3]) or classifiable_pos(part_of_speech):
                    mistagged = False
                    if part_of_speech!=raw_data[3]:
                        incorrect_pos_count += 1
                        incorrect_word_count += 1
                        mistagged = True
                    else:
                        corpus_tags = set()
                        for tag in raw_data[4].split(u'|'):
                            if tag!=u'_':
                                tx = tag.split(u'=')
                                tag_name = tx[0]
                                tag_value = tx[1]
                                corpus_tags.add( (tag_name,tag_value) )
                        if len(corpus_tags)!=len(tags):
                            incorrect_word_count += 1
                            mistagged = True
                        else:
                            for tag in tags:
                                if tag not in corpus_tags:
                                    incorrect_word_count += 1
                                    mistagged = True
                                    break
                    if mistagged:
                        wrt_result.write(
                            u'line={} word={} EXPECTED: {} {} MODEL: {} {}\n'.format(line_num, word,
                                                                                     raw_data[3],
                                                                                     raw_data[4],
                                                                                     part_of_speech,
                                                                                     tags_str))



                                    #if part_of_speech!=raw_data[3] or tags_str!=raw_data[4]:
                    #    incorrect_word_count += 1

            processed_sent_count += 1
            if (processed_sent_count%10)==0:
                if is_tagging:
                    print( '{0} sentences tagged so far'.format(processed_sent_count), end='\r' )
                else:
                    print( '{0} sentences evaluated so far, part_of_speech_err_rate={1} err_rate={2}'.format(processed_sent_count,incorrect_pos_count/float(total_word_count),incorrect_word_count/float(total_word_count)), end='\r' )

        good = True
        sent = []
        if is_tagging:
            wrt_result.write('\n')  # пустая строка между предложениями в корпусе
    else:
        tx = line.split(u'\t')
        if len(tx)<2:
            good = False
        elif is_tagging:
            # в режиме разметки входных данных нам нужны только 2 столбца - номер токена и слова.
            # осатальные столбцы могут отсутствовать.
            toknum = tx[0]
            word = normalize_word(tx[1]) # слово
            sent.append((word,(toknum,word,u'_',u'_',u'_')))
        else:
            # в режиме оценки входной корпус имеет все столбцы, как и приобучении.
            word = normalize_word(tx[1])
            sent.append( (word,tx) )

wrt_result.close()

print( 'Done, {0} sentences processed.'.format(processed_sent_count) )

if not is_tagging:
    print( 'part_of_speech err_rate={0}'.format( incorrect_pos_count/float(total_word_count) ) )
    print( 'total err_rate={0}'.format( incorrect_word_count/float(total_word_count) ) )
