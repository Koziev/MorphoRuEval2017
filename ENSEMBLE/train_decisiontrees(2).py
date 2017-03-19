# -*- coding: utf-8 -*-
'''
Эксперимент: обучение ансамбля классификаторов на базе decisiontree линейного регрессора
для определения части речи слова с учетом его контекста.

(c) Илья Козиев 2017 для morphorueval 2017 inkoziev@gmail.com

Обучение:
python train_decisiontrees(2).py входной_корпус
'''

from __future__ import print_function
import codecs
import numpy as np
import collections
import pickle
import sys
import json
import scipy.sparse
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.metrics
import unicodedata
import os.path


# ---------------------------------------------------------------------------

# Окончания слов становятся отдельными признаками для моделей.
# Для слова берутся окончания длиной от MIN до MAX
MIN_SUFFIX_LEN = 1
MAX_SUFFIX_LEN = 4

# полный размер окна контекста
WINDOW=3

# Минимальная частота слова, которая делает его отдельным признаком
# в обучающем датасете.
FEATURED_WORD_MINFREQ = 5

# Добавлять ли номера кластеров слов в обучающий датасет.
# Кластеры слов получаются из word2vec встраиваний слов через процедуру кластеризации.
ADD_WORDCLUSTER_FEATURES = False

# Для отладки - можно ограничить размеры обучающего датасета.
MAX_PATTERNS = 1000000

# ---------------------------------------------------------------------------

# Условный токен для обозначения слов за границами предложения, когда
# окно модели находится рядом с границей.
BORDER_WORD = u'<nil>'

# ---------------------------------------------------------------------------

salient_pos = { u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET' }

word2cluster = dict()
featured_words = set()

def extract_features(word, feature_name_prefix):
    features = []

    if word == BORDER_WORD:
        features.append(feature_name_prefix + u'<nil>')
    else:
        lword = word.lower()
        if word[0].isupper() and word[1:].islower():
            features.append(feature_name_prefix + u'Aa')
        elif word.isupper():
            features.append(feature_name_prefix + u'AA')

        wlen = len(word)
        if wlen > 2:
            for suffix_len in range(MIN_SUFFIX_LEN, min(MAX_SUFFIX_LEN, wlen - 1) + 1):
                suffix = lword[wlen - suffix_len: wlen]
                features.append(feature_name_prefix + u'sfx=' + suffix)
                # print( u'DEBUG word={0} suffix_len={1} suffix={2}'.format(word,suffix_len,suffix) )
                # raw_input( 'press a key' )
        else:
            features.append(feature_name_prefix + u'word=' + lword)

        if lword in word2cluster:
            features.append(feature_name_prefix + u'cluster=' + word2cluster[lword])

        if lword in featured_words:
            features.append(feature_name_prefix + u'featured_word=' + lword)

    return features


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


def is_punct(word):
    return unicodedata.category(word[0]) == 'Po'

# ---------------------------------------------------------------------------

corpus_path = sys.argv[1]
model_folder = sys.argv[-1]

salient_tags = set()
pos2id = dict()
corpus = []
word2freq = collections.Counter()
pos2tags = { u'???':set() }
for pos in salient_pos:
    pos2tags[pos] = set()

print( 'Analyzing corpus {}...'.format(corpus_path) )

with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
    sent = []
    good = True
    for line0 in rdr:
        line = line0.strip()
        if len(line) == 0:
            if good and len(sent) > 0:
                corpus.append(sent)

                for word,pos,tags in sent:
                    word2freq[word] += 1
            good = True
            sent = []
        else:
            tx = line.split('\t')
            if len(tx) < 2:
                good = False
            else:
                word = normalize_word(tx[1])
                part_of_speech = tx[3]  # часть речи
                tags_str = tx[4].strip()

                if part_of_speech not in salient_pos:
                    part_of_speech = u'???'
                    tags_str = u'_'

                if part_of_speech not in pos2id:
                    pos2id[part_of_speech] = len(pos2id)

                tags = [] if tags_str==u'_' else tags_str.split(u'|')
                sent.append((word,part_of_speech,tags))
                if part_of_speech in pos2tags:
                    pos2tags[part_of_speech].update(tags)
                    salient_tags.update(tags)

print('Done, corpus.count={}'.format(len(corpus)))

# --------------------------------------------------------------------

with codecs.open( os.path.join(model_folder,'dt_pos2id.dat'), 'w', 'utf-8') as wrt:
    for pos,id in pos2id.iteritems():
        wrt.write( u'{0}\t{1}\n'.format(pos,id) )


# Теги, которые употребляются со значимыми частями речи, мы соберали в список salient_tag_names.
# Для каждого тэга в этом списке мы соберем список его значений и сохраним списки в словаре salient_tag_values
# В дальнейшем для каждого тэга будем строить отдельный классификатор по всем возможным значениям.

# сохраняем для каждой части речи список тегов, который употребляются с ней в корпусе,
# чтобы при распознавании запускать только нужные классификаторы.
#with codecs.open( 'dt_pos2tags.dat', 'w', 'utf-8') as wrt:
#    for pos,tags in pos2tags.iteritems():
#        wrt.write( u'{0}\t{1}\n'.format(pos, unicode.join( u' ', tags)) )

salient_tag_names = set( [ p.split(u'=')[0] for p in salient_tags ] )

salient_tag_values = dict()
for salient_tag_name in salient_tag_names:
    salient_tag_values[salient_tag_name] = set()

for tag in salient_tags:
    tx = tag.split(u'=')
    salient_tag_values[tx[0]].add(tx[1])

print( 'Storing {0} salient tag names...'.format(len(salient_tag_names)))
tagname2id = dict( [ (tag,i) for i,tag in enumerate(salient_tag_names) ] )
with codecs.open( os.path.join(model_folder,'dt_salient_tags.dat'), 'w', 'utf-8') as wrt:
    for tag_name, tag_values in salient_tag_values.iteritems():
        t_id = tagname2id[tag_name]
        wrt.write( u'{0}\t{1}\t{2}\n'.format(t_id,tag_name, unicode.join( u' ', tag_values ) ) )

# --------------------------------------------------------------------

# Список значимых слов получается из частотного словаря применением минимальной частоты.
# Таким образом, большинство служебных слов (предлоги, связочные глаголы и так далее) будут
# фигурировать в датасете как отдельные признаки.
featured_words = set( [ w for w,cnt in word2freq.iteritems() if cnt>FEATURED_WORD_MINFREQ ] )
print( 'Storing {0} featured words...'.format(len(featured_words)) )
with codecs.open( os.path.join(model_folder,'dt_featured_words.dat'), 'w', 'utf-8' ) as wrt:
    for word in featured_words:
        wrt.write( u'{0}\n'.format(word))

# --------------------------------------------------------------------

if ADD_WORDCLUSTER_FEATURES==True:
    with codecs.open( os.path.join(model_folder,'word2cluster.dat'),'r','utf-8') as rdr:
        print( 'Loading word2cluster...' )
        for line in rdr:
            parts = line.strip().split(u'\t')
            word = parts[0].lower()
            word2cluster[word] = parts[1]

# --------------------------------------------------------------------

feature2id = dict()

patterns = []
pattern_tags = []

for sent in corpus:
    nword = len(sent)
    for iword in range(nword):
        features = []
        part_of_speech = sent[iword][1]
        tags = sent[iword][2]

        if part_of_speech in pos2id:
            for j in range(WINDOW):
                jword = iword-(WINDOW-1)/2+j
                word = BORDER_WORD
                if jword>=0 and jword<nword:
                    word = sent[jword][0]
                prefix = u'[{}]'.format(-(WINDOW-1)/2+j)
                features.extend( extract_features(word,prefix) )
            feature_ids = set()
            for feature in features:
                if feature not in feature2id:
                    feature2id[feature]=len(feature2id)
                feature_ids.add( feature2id[feature] )

            y = pos2id[part_of_speech]
            patterns.append( (feature_ids,y) )
            pattern_tags.append( tags )

n_pattern = len(patterns)
nb_features = len(feature2id)
nb_labels = len(pos2id)

print( 'n_pattern={}'.format(n_pattern))
print( 'nb_features={}'.format(nb_features) )
print( 'nb_labels={}'.format(nb_labels) )

with codecs.open( os.path.join(model_folder,'dt_feature2id.dat'), 'w', 'utf-8') as wrt:
    for feature,f_id in feature2id.iteritems():
        wrt.write( u'{0}\t{1}\n'.format(feature,f_id) )

# ---------------------------------------------------------------------------

# сохраним метапараметры в config файл, чтобы тэггер знал как формировать датасет.
with codecs.open(os.path.join(model_folder,'dt_postagger.config'), 'w', 'utf-8') as cfg:
    params = {
        'WINDOW': WINDOW,
        'MIN_SUFFIX_LEN': MIN_SUFFIX_LEN,
        'MAX_SUFFIX_LEN': MAX_SUFFIX_LEN,
        # 'LOWERCASE': LOWERCASE,
        'FEATURED_WORD_MINFREQ': FEATURED_WORD_MINFREQ,
        'ADD_WORDCLUSTER_FEATURES': ADD_WORDCLUSTER_FEATURES,
        'BORDER_WORD': BORDER_WORD,
        'nb_features':nb_features,
        'nb_labels':nb_labels # для классификатора частей речи
    }
    json.dump(params, cfg)

# ---------------------------------------------------------------------------------

n_pattern = min( MAX_PATTERNS, n_pattern )

split = 0.1

n_test = int(split * n_pattern)
n_train = n_pattern - n_test

print('n_train={0} n_test={1}'.format(n_train, n_test))

# -----------------------------------------------------------------

# Готовим sparse матрицу данных для обучения классификаторов

X_train = scipy.sparse.lil_matrix((n_train, nb_features))
#X_train = np.zeros( (n_train,nb_features), dtype=np.bool)
y_train = np.zeros((n_train), dtype=np.int32)

X_test = scipy.sparse.lil_matrix((n_test, nb_features))
#X_test = np.zeros( (n_test,nb_features), dtype=np.bool)
y_test = np.zeros((n_test), dtype=np.int32)

itest = 0
itrain = 0
for features,y in patterns[0:n_pattern]:

        if itest < n_test:
            y_test[itest] = y

            for feature_id in features:
                X_test[ itest, feature_id ] = True

            itest += 1
        else:
            y_train[itrain] = y

            for feature_id in features:
                X_train[itrain, feature_id ] = True

            itrain += 1

# -----------------------------------------------------------------

print( '\n\nFitting MultinomialNB...')
cl = sklearn.naive_bayes.MultinomialNB()
cl.fit(X_train,y_train)
print('Estimating the accuracy of classifier...')
acc = cl.score(X_test,y_test)
print( 'acc={}'.format(acc))

# todo - вывести important features


# -----------------------------------------------------------------
print( '\n\nFitting LogisticRegression...')
cl = sklearn.linear_model.LogisticRegression( penalty='l2',
                                              dual=False,
                                              tol=0.0001,
                                              C=1.0,
                                              fit_intercept=True,
                                              intercept_scaling=1,
                                              class_weight=None,
                                              random_state=None,
                                              solver='liblinear',
                                              max_iter=100,
                                              multi_class='ovr',
                                              verbose=0,
                                              warm_start=False,
                                              n_jobs=1)
cl.fit(X_train,y_train)
print('Estimating the accuracy of classifier...')
acc = cl.score(X_test,y_test)
print( 'acc={}'.format(acc))

with open( os.path.join(model_folder,'LogisticRegression.pickle'), 'wb') as f:
    pickle.dump(cl,f)


# ---------------------------------------------------------------------

print('\n\nFitting DecisionTreeClassifier...')
cl = sklearn.tree.DecisionTreeClassifier(criterion='gini',
                                         splitter='best',
                                         max_depth=None,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0,
                                         max_features=None,
                                         random_state=None,
                                         max_leaf_nodes=None,
                                         #min_impurity_split=1e-07,
                                         #class_weight=None,
                                         presort=False)

cl.fit(X_train, y_train)

print('Estimating...')
acc = cl.score(X_test,y_test)
print( 'acc={}'.format(acc))

with open( os.path.join(model_folder,'DecisionTreeClassifier.pickle'), 'wb') as f:
    pickle.dump(cl,f)

# ---------------------------------------------------------------------

# print('\n\nFitting RandomForestClassifier...')
# cl = sklearn.ensemble.RandomForestClassifier(criterion='gini',
#                               n_estimators=10,
#                               min_samples_split=2,
#                               min_samples_leaf=1,
#                               max_features='auto',
#                               oob_score=False,
#                               random_state=1,
#                               n_jobs=1,
#                               verbose=0)
# cl.fit(X_train, y_train)
# print('Estimating...')
# acc = cl.score(X_test,y_test)
# print( 'acc={}'.format(acc))

# --------------------------------------------------------------------------

#print( '\n\nFitting GradientBoostingClassifier...' )
#cl = sklearn.ensemble.GradientBoostingClassifier(n_estimators=10)
#cl.fit( X_train, y_train )
#print( 'Estimating...' )
#acc = cl.score(X_test,y_test)
#print( 'acc={}'.format(acc))

# ----------------------------------------------------------------------------

# теперь цикл подготовки датасета и обучения классификаторов для каждого значимого тэга

codebook = dict()

for salient_tag in salient_tag_names:

    nvalue = len(salient_tag_values[salient_tag])
    tag_id = tagname2id[salient_tag]
    print( '\nTraining for detection of tag {0} ({1} values: {2})'.format(salient_tag,nvalue,unicode.join(u' ',salient_tag_values[salient_tag])) )

    # ключ 0 оставляем за вариантом "нет никакого значения тега TTT"
    value2id = dict( [(v,i+1) for i,v in enumerate(salient_tag_values[salient_tag])] )
    
    codebook[salient_tag] = (tag_id,value2id)

    itest = 0
    itrain = 0
    n_positive=0

    for ipattern,tags in enumerate(pattern_tags[0:n_pattern]):
        y = 0
        for tag in tags:
            tx = tag.split(u'=')
            if tx[0]==salient_tag:
                y = value2id[tx[1]]
                n_positive += 1
                break

        if itest < n_test:
            y_test[itest] = y
            itest += 1
        else:
            y_train[itrain] = y
            itrain += 1

    print( u'Fitting LogisticRegression for {0} with {1} positive samples...'.format(salient_tag,n_positive))
    cl = sklearn.linear_model.LogisticRegression( penalty='l2',
                                                  dual=False,
                                                  tol=0.0001,
                                                  C=1.0,
                                                  fit_intercept=True,
                                                  intercept_scaling=1,
                                                  class_weight=None,
                                                  random_state=None,
                                                  solver='liblinear',
                                                  max_iter=100,
                                                  multi_class='ovr',
                                                  verbose=0,
                                                  warm_start=False,
                                                  n_jobs=1)
    cl.fit(X_train,y_train)
    acc = cl.score(X_test,y_test)

    y_pred = cl.predict(X_test)
    f1 = sklearn.metrics.f1_score( y_test, y_pred, average='weighted' )

    print( 'acc={} f1_score={}'.format(acc,f1))

    with open( os.path.join(model_folder,'LogisticRegression_{0}.pickle').format(tag_id), 'wb') as f:
        pickle.dump(cl,f)

    # -------------------------------------------------------------------

    print( u'Fitting DecisionTreeClassifier for {0} with {1} positive samples...'.format(salient_tag,n_positive))
    cl = sklearn.tree.DecisionTreeClassifier(criterion='gini',
                                             splitter='best',
                                             max_depth=None,
                                             min_samples_split=2,
                                             min_samples_leaf=1,
                                             min_weight_fraction_leaf=0.0,
                                             max_features=None,
                                             random_state=None,
                                             max_leaf_nodes=None,
                                             #min_impurity_split=1e-07,
                                             #class_weight=None,
                                             presort=False)
    cl.fit(X_train,y_train)
    acc = cl.score(X_test,y_test)

    y_pred = cl.predict(X_test)
    f1 = sklearn.metrics.f1_score( y_test, y_pred, average='weighted' )

    print( 'acc={} f1_score={}'.format(acc,f1))

    with open( os.path.join(model_folder,'DecisionTreeClassifier_{0}.pickle').format(tag_id), 'wb') as f:
        pickle.dump(cl,f)
        
with codecs.open( os.path.join(model_folder,'dt_codebook.dat'), 'w', 'utf-8' ) as wrt:
    for tag_name,tag_data in sorted( codebook.iteritems(), key=lambda z:z[1][0] ):
        tag_id = tag_data[0]
        value2id = tag_data[1]
        wrt.write( u'{0}\t{1}\t'.format(tag_name,tag_id) )
        wrt.write( u'{0}\n'.format( unicode.join( u' ', [ unicode(i)+u':'+v for (v,i) in value2id.iteritems()] ) ) )

            
