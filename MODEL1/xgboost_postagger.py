# -*- coding: utf-8 -*-
'''
POS Tagger на базе символьной модели и классификатора xgboost
(c) Илья Козиев 2017 для morphorueval 2017 inkoziev@gmail.com

Обучение:
python xgboost_postagger.py learn ./CORPORA/GIKRYA_texts.txt model_data_folder

Разметка:
python xgboost_postagger.py tag imput_corpora output_corpora model_data_folder

Оценка модели по размеченному корпусу:
python xgboost_postagger.py eval ./CORPORA/morpheval_corpus_solarix.dat model_data_folder

'''

from __future__ import print_function
import codecs
import collections
import json
import sys
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
import unicodedata

# Полный размер окна (распознаваемое слово входит в него)
WINDOW = 7

# Использовать обратный порядок символов в слове.
# Этот прием в данной задаче сильно улучшает результаты сетки, так как
# морфологические признаки слова в подавляющем числе случаев связаны с его
# окончанием и суффиксом. Если рассматривать символы слова справа налево, то
# окончание всегда оказывается на одних и тех же входах сетки, что улучшает
# сходимость бэкпропа при малом кол-ве слоев, а это актуально для небольшого
# обучающего датасета в наших случаях.
INVERT = True

run_mode = ''  # learn|tag|eval

# ----------------------------------------------------------------------

classifiable_posx = {u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET'}


def classifiable_pos(part_of_speech):
    return True if part_of_speech in classifiable_posx else False


def is_punct(word):
    return unicodedata.category(word[0]) == 'Po'


def canonize_tagset(tagset_str):
    return unicode.join(u'|', sorted(tagset_str.split(u'|')))


# Вернет число в диапазоне 0...1, характеризующее степень точность представления тегсета корпуса
# тегсетом модели.
def eq_tagsets(model_tagset, corpus_tagset):
    s1 = set(model_tagset.split(u'|'))
    s2 = set(corpus_tagset.split(u'|'))
    s12 = s1 + s2
    num = sum([1 for t in s2 if t in s1])
    return num / float(len(s12))


def normalize_word(word):
    return word.replace(u'ё', u'е').lower()


valid_chars = set([c for c in u'abcdefghijklmnopqrstuvwxyz0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюя'])


def normalize_char(c):
    if c in valid_chars:
        return c
    elif unicodedata.category(c) == 'Po':
        return c
    else:
        return u' '


def word_chars(word, invert):
    return [normalize_char(c) for c in (word[::-1] if invert else word)]


def usage():
    print('Usage:')
    print('python xgboost_postagger.py learn input_corpus model_data_folder')
    print('python xgboost_postagger.py tag input_corpus output_file model_data_folder')
    print('python xgboost_postagger.py eval input_corpus output_file model_data_folder')
    exit()


# ---------------------------------------------------------------------

if len(sys.argv) < 4:
    usage()

run_mode = sys.argv[1]
corpus_path = sys.argv[2]
model_folder = sys.argv[-1]

# ----------------------------------------------------------------------
special_word2pos = dict()

with codecs.open('special_words.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        px = line.strip().split(u'\t')
        word = px[0]
        pos = px[1]
        special_word2pos[word] = pos


def is_classifiable_word(word):
    return word not in special_word2pos and not is_punct(word)


def eq_tagsets(tagset1, tagset2):
    if len(tagset1) != len(tagset2):
        return False
    return True if set(tagset1) == set(tagset2) else False


# ----------------------------------------------------------------------

if run_mode == 'learn':

    # Загрузим тренировочный датасет.
    print('Loading "{0}...'.format(corpus_path))
    corpus = []
    ntoken = 0
    n_bad = 0
    sent_count = 0
    # Если обучающий датасет не влезет в память, то можно ограничить число паттернов в нем
    max_patterns = 1000000

    word2freq = collections.Counter()
    pos2index = dict()

    with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
        sent = []
        good = True
        for line0 in rdr:
            line = line0.strip()
            if len(line) == 0:
                if good and len(sent) > 0:
                    corpus.append(sent)
                    ntoken += len(sent)
                else:
                    n_bad += 1
                good = True
                sent = []
            else:
                tx = line.split('\t')
                if len(tx) < 2:
                    good = False
                else:
                    word = normalize_word(tx[1].lower())  # слово
                    word2freq[word] += 1
                    tagset = u''
                    pos = tx[3]  # часть речи

                    if is_classifiable_word(word):
                        if classifiable_pos(pos):
                            features = tx[4] if len(tx) == 5 else u''  # теги
                            tagset = (pos + u' ' + canonize_tagset(features)).strip()
                        else:
                            tagset = u'???'

                    sent.append((word, tagset))

    print('Done, {0} sentences, ntoken={1}'.format(len(corpus), ntoken))

    n_patterns = 0
    all_chars = set()
    word2tag = dict()
    max_word_len = 0
    for sent in corpus:
        for word_data in sent:
            word = word_data[0]
            max_word_len = max(max_word_len, len(word))
            all_chars.update(word_chars(word, INVERT))
            tagset = word_data[1]
            if tagset != u'':
                n_patterns += 1
                if tagset not in pos2index:
                    pos2index[tagset] = len(pos2index)

                id_tagset = pos2index[tagset]
                if word not in word2tag:
                    word2tag[word] = id_tagset
                elif word2tag[word] != id_tagset:
                    word2tag[word] = -1

    output_size = len(pos2index)
    print('output_size={}'.format(output_size))

    char2index = { u'':0 }
    for i,c in enumerate(all_chars):
        char2index[c] = i+1

    print( 'number of chars={}'.format(len(all_chars)) )
    print( 'max_word_len={}'.format(max_word_len) )

    with codecs.open(os.path.join(model_folder, 'm2_char2index.dat'), 'w', 'utf-8') as wrt:
        for ch, id in char2index.iteritems():
            wrt.write(u'{0}\t{1}\n'.format(ch, id))

    with codecs.open(os.path.join(model_folder, 'm2_word2tag.dat'), 'w', 'utf-8') as wrt:
        for word, tag in word2tag.iteritems():
            if tag != -1:
                wrt.write(u'{0}\t{1}\n'.format(word, tag))

    with codecs.open(os.path.join(model_folder, 'm2_word2freq.dat'), 'w', 'utf-8') as wrt:
        for word, cnt in sorted(word2freq.iteritems(), key=lambda z: -z[1]):
            wrt.write(u'{0}\t{1}\n'.format(word, cnt))

    with codecs.open(os.path.join(model_folder, 'm2_id2tag.dat'), 'w', 'utf-8') as wrt:
        for tag, id in pos2index.iteritems():
            wrt.write(u'{0}\t{1}\n'.format(id, tag))

    # ----------------------------------------------------------------------

    n_patterns = min(max_patterns, n_patterns)
    print('n_patterns={0}'.format(n_patterns))

    # ----------------------------------------------------------------------

    print('Vectorization...')

    input_size = WINDOW*max_word_len
    winspan = int((WINDOW - 1) / 2)

    X_data = np.zeros((n_patterns, input_size ), dtype=np.int32)
    y_data = np.zeros((n_patterns), dtype=np.int32)
    idata = 0
    for sent in corpus:
        if idata == n_patterns:
            break
        nword = len(sent)
        for iword in range(nword):
            pos = sent[iword][1]
            if pos != u'':
                for j in range(WINDOW):
                    word_index = iword - winspan + j
                    word = u''
                    if word_index >= 0 and word_index < nword:
                        word = sent[word_index][0]
                    for ichar, c in enumerate(word_chars(word, INVERT)):
                        bit_index = j*max_word_len + ichar
                        X_data[idata, bit_index] = char2index[c]
                y_data[idata] = pos2index[pos]
                idata += 1
                if idata == n_patterns:
                    break

    print( 'Preparing xgboost datasets...' )
    X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.1 )
    D_train = xgboost.DMatrix(X_train, label=y_train)
    D_test = xgboost.DMatrix(X_test, label=y_test)

    xgb_params = {
        'booster': 'gbtree',  # 'dart',
        'n_estimators': 100,
        'subsample': 0.5,
        'max_depth': 10,
        'eta': 0.05,
        'scale_pos_weight': 1,
        'silent': 1,
        #'eval_metric': 'logloss',
        'objective': 'multi:softmax',
        'num_class': output_size
    }

    print( 'Training xgboost...' )

    cl = xgboost.train(xgb_params, D_train,
                   num_boost_round=1000,
                   evals=[(D_test, 'eval')],
                   early_stopping_rounds=20,
                   verbose_eval=True
                   )

    ntree_limit = cl.best_ntree_limit
    print( 'xgboost completed, ntree_limit={}'.format(ntree_limit) )

    cl.save_model(os.path.join(model_folder, 'm2_xgboost.model'))

    with codecs.open(os.path.join(model_folder, 'm2_xgboost_postagger.config'), 'w', 'utf-8') as cfg:
        params = {
            'WINDOW': WINDOW,
            'max_word_len': max_word_len,
            'output_size': output_size,
            'invert': INVERT,
            'ntree_limit': ntree_limit
        }
        json.dump(params, cfg)


# --------------------------------------------------------------------------------------------

elif run_mode == 'tag' or run_mode == 'eval':
    UNAMBIG_FREQ_THRESHOLD = 2
    word2freq = dict()
    word2tag = dict()
    id2tag = dict()
    char2index = dict()

    is_tagging = True
    if run_mode == 'eval':
        is_tagging = False

    if len(sys.argv) != 5:
        usage()

    input_path = sys.argv[2]
    result_path = sys.argv[3]

    if is_tagging:
        print(u'Tagging data in {0} and writing results to {1}...'.format(input_path, result_path))
    else:
        print(u'Model evaluation on {}, validation results will be written to {}...'.format(input_path, result_path))

    with codecs.open(os.path.join(model_folder, 'm2_char2index.dat'), 'r', 'utf-8') as rdr:
        for line in rdr:
            parts = line.split(u'\t')
            if len(parts) == 2:
                char2index[parts[0]] = int(parts[1].strip())

    with codecs.open(os.path.join(model_folder, 'm2_id2tag.dat'), 'r', 'utf-8') as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            if len(parts) == 2:
                id2tag[int(parts[0])] = parts[1]

    with codecs.open(os.path.join(model_folder, 'm2_word2freq.dat'), 'r', 'utf-8') as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            if len(parts) == 2:
                word2freq[parts[0]] = int(parts[1])

    with codecs.open(os.path.join(model_folder, 'm2_word2tag.dat'), 'r', 'utf-8') as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            if len(parts) == 2:
                freq = int(parts[1])
                if freq >= UNAMBIG_FREQ_THRESHOLD:
                    word2tag[parts[0]] = freq

    with open(os.path.join(model_folder, 'm2_xgboost_postagger.config'), 'rt') as cfg:
        params = json.load(cfg)
        WINDOW = int(params['WINDOW'])
        max_word_len = int(params['max_word_len'])
        output_size = int(params['output_size'])
        INVERT = bool(params['invert'])

    booster = xgboost.Booster()
    booster.load_model( os.path.join(model_folder, 'm2_xgboost.model') )

    rdr_corpus = codecs.open(input_path, 'r', 'utf-8')

    wrt_result = codecs.open(result_path, 'w', 'utf-8')

    winspan = int((WINDOW - 1) / 2)
    
    input_size = WINDOW*max_word_len

    processed_sent_count = 0
    total_word_count = 0
    incorrect_pos_count = 0
    incorrect_tagset_count = 0
    sent = []
    good = True
    line_num = 0
    for line0 in rdr_corpus:
        line = line0.strip()
        line_num += 1
        if len(line) == 0:
            if good and len(sent) > 0:

                n_patterns = len(sent)

                # Пропустим через сеточный классификатор паттерны для всех слов
                # этого предложения. Можно оптимизировать алгоритм, если подавать
                # на вход только те слова, для которых нет мемоизированного значения.
                X_data = np.zeros((n_patterns, WINDOW*max_word_len), dtype=np.int32)
                idata = 0
                nword = len(sent)
                for iword in range(nword):
                    for j in range(WINDOW):
                        word_index = iword - winspan + j
                        word = u''
                        if word_index >= 0 and word_index < nword:
                            word = sent[word_index][0]
                        for ichar, c in enumerate(word_chars(word, INVERT)):
                            if c in char2index:
                                bit_index =  j*max_word_len + ichar
                                X_data[idata, bit_index] = char2index[c]
                    idata += 1

                D_data = xgboost.DMatrix(X_data)
                y = booster.predict(D_data)
                for iword in range(nword):
                    word = sent[iword][0]
                    raw_data = sent[iword][1]

                    part_of_speech = u'X'
                    tags = u'_'

                    if word in special_word2pos:
                        part_of_speech = special_word2pos[word]
                        tags = u'_'
                    elif is_punct(word):
                        part_of_speech = u'PUNCT'
                        tags = u'_'
                    else:
                        tag_id = -1
                        if word in word2tag:
                            tag_id = word2tag[word]
                        else:
                            tag_id = int(y[iword])

                        tagset_parts = id2tag[tag_id].split(u' ')
                        part_of_speech = tagset_parts[0]
                        tags = u'' if len(tagset_parts) == 1 else tagset_parts[1]

                    if is_tagging:
                        total_word_count += 1
                        wrt_result.write(
                            u'{0}\t{1}\t{2}\t{3}\t{4}\n'.format(raw_data[0], raw_data[1], raw_data[2], part_of_speech,
                                                                tags))
                    else:
                        corpus_pos = raw_data[3]
                        corpus_tags = raw_data[4]

                        if word not in special_word2pos and (
                            classifiable_pos(part_of_speech) or classifiable_pos(corpus_pos)):
                            total_word_count += 1
                            if part_of_speech != corpus_pos:
                                incorrect_pos_count += 1
                                incorrect_tagset_count += 1
                            else:
                                incorrect_tagset_count += (1.0 - eq_tagsets(tags, corpus_tags))

                            if part_of_speech != corpus_pos or not eq_tagsets(tags, corpus_tags):
                                wrt_result.write(
                                    u'line_num={} word={} required pos={} model_pos={} required_tags={} model_tags={}\n'.format(
                                        line_num, word, corpus_pos, part_of_speech, corpus_tags, tags))

                if is_tagging:
                    wrt_result.write('\n')  # пустая строка между предложениями в корпусе

                processed_sent_count += 1
                if (processed_sent_count % 100) == 0:
                    if is_tagging:
                        print('{0} sentences have been tagged so far'.format(processed_sent_count), end='\r')
                    else:
                        part_of_speech_err_rate = incorrect_pos_count / float(total_word_count)
                        tagset_err_rate = incorrect_tagset_count / float(total_word_count)
                        print(
                            '{0} sentences have been evaluated so far, part_of_speech_err_rate={1} tagset_err_rate={2}'.format(
                                processed_sent_count, part_of_speech_err_rate, tagset_err_rate), end='\r')

            good = True
            sent = []
        else:
            tx = line.split(u'\t')
            if len(tx) < 2:
                good = False
            elif is_tagging:
                # в режиме разметки входных данных нам нужны только 2 столбца - номер токена и слова.
                # осатальные столбцы могут отсутствовать.
                toknum = tx[0]
                word = normalize_word(tx[1]) # слово
                sent.append((word,(toknum,word,u'_',u'_',u'_')))
            else:
                # в режиме оценки входной корпус имеет все столбцы, как и приобучении.
                word = normalize_word(tx[1]) # слово
                sent.append( (word,tx) )

    print('\nDone, {0} sentences processed.'.format(processed_sent_count))

    wrt_result.close()

    if not is_tagging:
        part_of_speech_err_rate = incorrect_pos_count / float(total_word_count)
        tagset_err_rate = incorrect_tagset_count / float(total_word_count)
        print('part_of_speech_err_rate={0} tagset_err_rate={1}'.format(part_of_speech_err_rate, tagset_err_rate),
              end='\r')

# ------------------------------------------------------------------------
else:
    print('Unknown scenario {0}'.format(run_mode))
    usage()
