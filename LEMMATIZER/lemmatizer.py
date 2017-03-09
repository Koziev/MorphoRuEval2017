# -*- coding: utf-8 -*- 
'''
Лемматизатор на базе символьной сеточной модели.
(c) Илья Козиев 2017 для morphorueval 2017 inkoziev@gmail.com

Сценарии использования:

Обучение:
python lemmatizer.py learn ../CORPORA/GIKRYA_texts.txt  каталог_для_файлов_модели

Лемматизация:
python lemmatizer.py tag1 входной_корпус  выходной_файл  каталог_с_файлами_модели

команды tag1 и tag2 включают разметку без использования стороннего словаря Solarix (закрытая дорожка)
или с использованием (открытая дорожка) соответственно.

Оценка модели по размеченному корпусу:
python lemmatizer.py eval1 ./CORPORA/morpheval_corpus_solarix.dat  файл_для_сохранения_ошибок  каталог_с_файлами_модели

Различия между командами eval1 и eval2 аналогичны различиям между tag1 и tag2, описанным выше.
При оценке проверяется совпадение леммы, полученной моделью, и леммы в указаном корпусе. Регистр букв
и е/ё не различаются. Подсчитываются только слова, относящиеся к классифицируемым категориям согласно
условиям конкурса.

'''

from __future__ import print_function
import codecs
import collections
import json
import sys
import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.regularizers import l1, l2
import os.path
import unicodedata

run_mode = '' # learn|tag1|tag2|eval1|eval2

# Сеточная модель работает ощутимо точнее, если символы в словах рассматривать
# в порядке справа налево. При этом последние символы поступают на вход нейросети
# в фиксированных позициях, что облегчает настройку весов в условиях малого количества
# обучающих сэмплов.
INVERT = True

# Слова с частотой не менее указанной, у которых найдена однозначная лемма, будут обработаны
# не сеточной моделью, а по собранному в ходе стат. анализа корпуса словарю.
UNAMBIG_FREQ_THRESHOLD = 2

# Для дорожки с обучением по своему корпусу - используем лемматизацию из словаря Solarix
USE_LEXICON = False

# ----------------------------------------------------------------------


def is_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def normalize_word(word):
    return word.replace(u'ё',u'е')


def eq_lemmas( lemma1, lemma2 ):
    return normalize_word(lemma1.lower()) == normalize_word(lemma2.lower())


def reset_lemma_Aa( word, lemma, part_of_speech, itoken ):
    if itoken>0 and part_of_speech==u'NOUN' and word[0].isupper() and word[1:].islower():
        return lemma[0].upper() + lemma[1:]
    else:
        return lemma



sol2ud = { u'СУЩЕСТВИТЕЛЬНОЕ':u'NOUN', u'ГЛАГОЛ':u'VERB', u'НАРЕЧИЕ':u'ADV', u'ПРИЛАГАТЕЛЬНОЕ':u'ADJ' }
def convert_to_ud_pos( pos ):
    if pos in sol2ud:
        return sol2ud[pos]
    else:
        return u''



classifiable_posx = { u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET' }
def classifiable_pos(part_of_speech):
    return True if part_of_speech in classifiable_posx else False



def usage():
    print( 'Usage:' )
    print( 'python lemmatizer.py learn input_corpus model_folder' )
    print( 'python lemmatizer.py tag input_corpus output_file model_folder' )
    print( 'python lemmatizer.py eval input_corpus model_folder' )
    exit()


if len(sys.argv)<4:
    usage()

run_mode = sys.argv[1]
corpus_path = sys.argv[2]
model_folder = sys.argv[-1]

if run_mode == 'eval1':
    USE_LEXICON = False
    run_mode = 'eval'
elif run_mode == 'eval2':
    USE_LEXICON = True
    run_mode = 'eval'
elif run_mode == 'tag1':
    USE_LEXICON = False
    run_mode = 'tag'
elif run_mode == 'tag2':
    USE_LEXICON = True
    run_mode = 'tag'


# ----------------------------------------------------------------------

if run_mode == 'learn':
    # Загрузим тренировочный датасет. 
    print( 'Loading "{0}"...'.format(corpus_path) )

    max_word_len = 0
    max_lemma_len = 0
    all_chars = set()
    word2freq = collections.Counter()
    pos2word2lemma = dict() # справочник слов, которые имеют однозначную лемматизацию
    pos2word2lemmas = dict() # справочник слов, которые имеют неоднозначную лемматизацию

    classifiable_posx = {u'NOUN', u'ADJ', u'VERB', u'PRON', u'ADV', u'NUM', u'DET'}
    for pos in classifiable_posx:
        pos2word2lemma[ pos ] = dict()
        pos2word2lemmas[ pos ] = dict()

    tags_set = set()
    
    word_dataset = set()
    
    with codecs.open( corpus_path, 'r', 'utf-8') as rdr:
        for line0 in rdr:
            line = line0.strip()
            if len(line)!=0:
                tx = line.split(u'\t')
                if len(tx)>=3:
                    word = normalize_word(tx[1].lower())
                    lemma = normalize_word(tx[2].lower())
                    pos = tx[3]
                    tags = tx[4]
                    
                    word2freq[word] += 1

                    if pos in pos2word2lemma and not is_int(word):
                        max_word_len = max(max_word_len, len(word))
                        max_lemma_len = max(max_lemma_len, len(lemma))
                        all_chars.update(word)

                        word2lemma = pos2word2lemma[pos]
                        word2lemmas = pos2word2lemmas[pos]
                        
                        if word in word2lemmas:
                            word2lemmas[word][lemma] += 1
                        else:
                            if word in word2lemma:
                                if word2lemma[word] != lemma:
                                    # оказалось, что это слов может иметь разную лемматизацию
                                    word2lemmas[word] = collections.Counter()
                                    word2lemmas[word][lemma] = 1 # новый вариант лемматизации
                                    word2lemmas[word][ word2lemma[word] ] = word2freq[word] # ранее найденный вариант лемматизации
                                    word2lemma[word] = u''
                            else:
                                word2lemma[word] = lemma
                            
                        tags_list = [ pos ]
                        tags_list.extend( tags.split(u'|') )
                        
                        tags_set.update( tags_list )
                        
                        d = (word,tuple(tags_list),lemma)
                        word_dataset.add( d )

    print( 'done, max_word_len={0} max_lemma_len={1}'.format(max_word_len,max_lemma_len) )
    for pos,word2lemma in pos2word2lemma.iteritems():
        print( u'{0} --> {1} unambigous lemmas'.format(pos,len(word2lemma)) ) 

    nb_tag = len(tags_set)
        
    bits_per_char = len(all_chars)
    input_size = nb_tag+max_word_len*bits_per_char
    output_size = max_lemma_len*bits_per_char
    print( 'bits_per_char={0}'.format(bits_per_char) )
    print( 'input_size={0}'.format(input_size) )

    char2index = dict([ (c,i) for (i,c) in enumerate(all_chars) ])
    tag2id = dict( [ (tag,i) for (i,tag) in enumerate(tags_set) ] )
    
    with codecs.open( os.path.join(model_folder,'lemmatizer_char2index.dat'), 'w', 'utf-8' ) as wrt:
        for ch,id in char2index.iteritems():
            wrt.write( u'{}\t{}\n'.format(ch,id) )

    with codecs.open( os.path.join(model_folder,'tag2id.dat'), 'w', 'utf-8' ) as wrt:
        for t,id in tag2id.iteritems():
            wrt.write( u'{}\t{}\n'.format(t,id) )

    with codecs.open( os.path.join(model_folder,'word2lemma.dat'), 'w', 'utf-8' ) as wrt:
        for pos,word2lemma in pos2word2lemma.iteritems():
            for word,lemma in word2lemma.iteritems():
                if lemma!=u'':
                    wrt.write( u'{}\t{}\t{}\n'.format(pos,word,lemma) )

    with codecs.open( os.path.join(model_folder,'word2lemmas.dat'), 'w', 'utf-8' ) as wrt:
        for pos,word2lemmas in pos2word2lemmas.iteritems():
            for word,lemmas in word2lemmas.iteritems():
                for lemma,freq in lemmas.iteritems():
                    wrt.write( u'{}\t{}\t{}\t{}\n'.format(pos,word,lemma,freq) )
                    
    with codecs.open( os.path.join(model_folder,'word2freq.dat'), 'w', 'utf-8' ) as wrt:
        for word,cnt in sorted( word2freq.iteritems(), key=lambda z:-z[1] ):
            wrt.write( u'{}\t{}\n'.format(word,cnt) )

    # ----------------------------------------------------------------------
    
    n_pattern = len(word_dataset)
    
    # найдем лемматизирующие трансдьюсеры
    transducer2id = dict()
    transducer2id[(0,u'')] = 0
    
    word_transducers = []
    for word,tags,lemma in word_dataset:
        stem_len = 0
        for i in range( min(len(word),len(lemma)) ):
            if word[i]!=lemma[i]:
                break
            stem_len += 1
            
        if stem_len>1:
            minus_ending_size = len(word)-stem_len
            plus_ending = lemma[stem_len:]
        else:
            minus_ending_size = 0
            plus_ending = u''

        t = (minus_ending_size,plus_ending)
        t_id = -1
        if t not in transducer2id:
            t_id = len(transducer2id)
            transducer2id[t] = t_id
        else:
            t_id = transducer2id[t]

        word_transducers.append( (word,tags,t_id) )

    print( u'{0} transducers in total'.format(len(transducer2id)) )

    output_size = len(transducer2id)
    print( u'Building net model for {0} patterns'.format( n_pattern ) )
    
    X_data = np.zeros( (n_pattern,input_size), dtype=np.bool )
    y_data = np.zeros( (n_pattern,output_size), dtype=np.bool )
    idata=0

    char_step = -1 if INVERT else 1
    for word,tags,t_id in word_transducers:
        for tag in tags:
            X_data[ idata, tag2id[tag] ] = True

        for ichar,ch in enumerate(word[::char_step]):
            X_data[ idata, nb_tag + ichar*bits_per_char+char2index[ch] ] = True

        y_data[ idata, t_id ] = True
        idata += 1    

    with codecs.open( os.path.join(model_folder,u'transducers.dat'), 'w', 'utf-8' ) as wrt:
        for t,id in transducer2id.iteritems():
            wrt.write( u'{0}\t{1}\t{2}\n'.format(t[0],t[1],id) )
        
    model = Sequential()
    HIDDEN_SIZE = output_size
    model.add( Dense( input_dim=input_size, output_dim=HIDDEN_SIZE, W_regularizer=l2(0.00001) ) )
    model.add( Dense( input_dim=input_size, output_dim=HIDDEN_SIZE, W_regularizer=l2(0.00001) ) )
    model.add( Dense( output_dim=output_size, activation='softmax', W_regularizer=l2(0.00001) ) )

    #opt = keras.optimizers.SGD( lr=0.1, momentum=0.1, decay=0, nesterov=False )
    opt = keras.optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #opt = keras.optimizers.Adagrad(lr=0.05, epsilon=1e-08)
    #opt='rmsprop'

    model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] )

    open( os.path.join(model_folder,u'lemmatizer.arch'), 'w' ).write( model.to_json() )

    model_checkpoint = ModelCheckpoint( os.path.join(model_folder,u'lemmatizer.model'), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping( monitor='val_acc', patience=5, verbose=1, mode='auto')

    history = model.fit( X_data, y_data, validation_split=0.1, batch_size=128, nb_epoch=100, callbacks=[model_checkpoint,early_stopping] )

    # ----------------------------------------------------------------------
    part_of_speech_list = [ pos for pos in pos2word2lemma ]
    with codecs.open( os.path.join(model_folder,'lemmatizer.config'),'w','utf-8') as cfg:
        params = { 
                  'max_word_len':max_word_len,
                  'bits_per_char':bits_per_char,
                  'input_size':input_size,
                  'output_size':output_size,
                  'nb_tag':nb_tag,
                  'part_of_speech_list':part_of_speech_list,
                  'invert':INVERT
                 }
        json.dump( params, cfg )

    print( 'Training finished.' )
# --------------------------------------------------------------------------------------------

elif run_mode == 'tag' or run_mode == 'eval':

    is_tagging = True
    if run_mode == 'eval':
        is_tagging = False
    
    if len(sys.argv)!=5:
        usage()
        
    input_path = sys.argv[2]
    result_path = sys.argv[3]

    if is_tagging:
        print( u'Tagging data in {0} and writing results to {1}...'.format(input_path,result_path) )
    else:    
        print( u'Model evaluation on {0}, errors will be stored in {1}'.format(input_path,result_path) )

    # для отладки - можно оценить вклад сеточного лемматизатора, отключив его флагом False.
    USE_NET_LEMMATIZATION = True

    lx_pos2word2lemma = dict()
    for pos in u'NOUN ADJ ADV VERB'.split(u' '):
        lx_pos2word2lemma[pos] = dict()

    if USE_LEXICON:
        lx_data = os.path.join(model_folder, 'lx_word2lemma.dat')
        print( u'Loading {0}...'.format(lx_data) )
        with codecs.open(lx_data, 'r', 'utf-8' ) as rdr:
            for line in rdr:
                px = line.strip().split(u'\t')
                word = normalize_word(px[0].lower())
                lemma = px[1]
                pos = px[2]
                ud_pos = convert_to_ud_pos(pos)
                if len(ud_pos)>0:
                    lx_pos2word2lemma[ud_pos][word] = lemma

    char2index = dict()
    with codecs.open( os.path.join(model_folder,'lemmatizer_char2index.dat'), 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.split( u'\t' )
            if len(parts)==2:
                char2index[parts[0]] = int(parts[1].strip())

    tag2id = dict()
    with codecs.open( os.path.join(model_folder,'tag2id.dat'), 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.strip().split( u'\t' )
            if len(parts) == 2:
                tag2id[parts[0]] = int(parts[1].strip())

    word2freq = dict()
    with codecs.open( os.path.join(model_folder,'word2freq.dat'), 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.strip().split( u'\t' )
            if len(parts) == 2:
                word2freq[ normalize_word(parts[0]) ] = int(parts[1])

    with open(os.path.join(model_folder,'lemmatizer.config'),'rt') as cfg:
        params = json.load( cfg )
        max_word_len = int(params['max_word_len'])
        bits_per_char = int(params['bits_per_char'])
        input_size = int(params['input_size'])
        output_size = int(params['output_size'])
        nb_tag = int(params['nb_tag'])
        part_of_speech_list = params['part_of_speech_list']
        INVERT = bool(params['invert'])

    pos2word2lemma = dict()
    pos2word2lemmas = dict()
    for pos in part_of_speech_list:
        pos2word2lemma[pos] = dict()
        pos2word2lemmas[pos] = dict()

    with codecs.open( os.path.join(model_folder,'word2lemma.dat'), 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            part_of_speech = parts[0]
            word = normalize_word(parts[1])
            lemma = parts[2]
            if word2freq[word]>=UNAMBIG_FREQ_THRESHOLD:
                pos2word2lemma[part_of_speech][word] = lemma

    with codecs.open( os.path.join(model_folder,'word2lemmas.dat'), 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            part_of_speech = parts[0]
            word = normalize_word(parts[1])
            lemma = parts[2]
            freq = int(parts[3])
            if word not in pos2word2lemmas[part_of_speech]:
                pos2word2lemmas[part_of_speech][word] = dict()
                
            pos2word2lemmas[part_of_speech][word][lemma] = freq
                
    model = model_from_json(open(os.path.join(model_folder,'lemmatizer.arch')).read())
    model.load_weights(os.path.join(model_folder,'lemmatizer.model'))

    id2transducer = dict()

    with codecs.open( os.path.join(model_folder,'transducers.dat'), 'r', 'utf-8' ) as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            minus_ending_size = int(parts[0])
            plus_ending = parts[1]
            t_id = int(parts[2])
            id2transducer[t_id] = (minus_ending_size,plus_ending)

    # все модели и трансдюсеры загружены.
    # читаем входной корпус

    print( u'Start processing {0}...'.format(input_path) )
    rdr_corpus = codecs.open( input_path, 'r', 'utf-8' )
    
    wrt_result = codecs.open( result_path, 'w', 'utf-8' )

    processed_sent_count = 0
    total_word_count = 0
    incorrect_word_count = 0
    sent = []
    good = True

    lemmatized_by_table = 0
    lemmatized_by_model = 0

    incorrect_lemmas = collections.Counter()

    while True:
        line = rdr_corpus.readline().strip()
        if len(line)==0:
            if len(sent)==0:
                break

            if good and len(sent)>0:

                n_patterns = len(sent)

                for iword,data in enumerate(sent):
                    word = data[0]
                    lemma = data[1]
                    part_of_speech = data[2]
                    tags = data[3]
                    raw_data = data[4]

                    loword = normalize_word(word.lower())
                    model_lemma = loword
                    lemmatized = False

                    #if loword==u'ее' and lemma==u'он' and part_of_speech==u'PRON':
                    #    print( 'DEBUG! ')

                    if part_of_speech in pos2word2lemma and not is_int(word):
                        word2lemma = pos2word2lemma[part_of_speech]
                        if loword in word2lemma:
                            model_lemma = reset_lemma_Aa( word, word2lemma[loword], part_of_speech, iword )
                            lemmatized_by_table += 1
                            lemmatized = True

                        if not lemmatized and USE_LEXICON:
                            if part_of_speech in lx_pos2word2lemma:
                                word2lemma = lx_pos2word2lemma[part_of_speech]
                                if loword in word2lemma:
                                    model_lemma = reset_lemma_Aa( word, word2lemma[loword], part_of_speech, iword )
                                    lemmatized_by_table += 1
                                    lemmatized = True

                        if not lemmatized and USE_NET_LEMMATIZATION:
                            tags_list = [ part_of_speech ]
                            tags_list.extend( tags )

                            X_data = np.zeros( (1,input_size), dtype=np.bool )

                            for tag in tags_list:
                                if tag in tag2id:
                                    X_data[0, tag2id[tag] ] = True

                            word_chars = loword[0:min(len(loword), max_word_len)]
                            if INVERT:
                                word_chars = word_chars[::-1]

                            for ichar,ch in enumerate(word_chars):
                                if ch in char2index:
                                    X_data[0, nb_tag+ichar*bits_per_char + char2index[ch] ] = True

                            y = model.predict_classes(X_data,verbose=0)
                            t_id = y[0]
                            transducer = id2transducer[t_id] # выбранный сеткой трансдьюсер
                            wlen = len(loword)
                            if wlen>transducer[0]+1:
                                model_lemma = loword[ 0:wlen-transducer[0] ] + transducer[1]
                                model_lemma = reset_lemma_Aa( word, model_lemma, part_of_speech, iword )
                                lemmatized_by_model += 1
                                lemmatized = True
                                #print( u'DEBUG sent={}'.format( unicode.join( u' ', [w[0] for w in sent] ) ) )
                                #print( u'DEBUG iword={} word={} generated lemma={}'.format(iword,word,model_lemma) )
                                #raw_input( 'DEBUG press a key' )
                                
                                if word in pos2word2lemmas[part_of_speech]:
                                    lemmas = pos2word2lemmas[part_of_speech][word]
                                    if model_lemma not in lemmas:
                                        # Предложенная моделью лемма не совпала ни с одним из вариантов лемм
                                        # для этого слова. Считаем, что модель выбрала неверный трансдьюсер.
                                        # Выберем лемму сэмплингом из частотной таблицы лемм для слова.
                                        sum_freq = sum( lemmas.itervalues() )
                                        lemma_probas = sorted( [ (freq/sum_freq,lemma) for (lemma,freq) in lemmas.iteritems() ], key=lambda z:-z[0] )
                                        sample_p = random.uniform(0.0,1.0)
                                        cumul_p = 0.0
                                        for (p,lemma) in lemma_probas:
                                            cumul_p += p
                                            if cumul_p >= sample_p:
                                                model_lemma = lemma
                                                break

                    if is_tagging:
                        total_word_count += 1
                        wrt_result.write( u'{0}\t{1}\t{2}\t{3}\t{4}\n'.format(raw_data[0],raw_data[1],model_lemma,raw_data[3],raw_data[4]) )
                    elif classifiable_pos(part_of_speech):
                        total_word_count += 1 # ошибку будем считать только по классифицируемым словам
                        if not eq_lemmas(lemma,model_lemma):
                            # не совпали леммы для классифицируемых по условиям конкурса частей речи.
                            incorrect_word_count+=1
                            error_key = (loword,model_lemma.lower(),lemma.lower(),part_of_speech)
                            incorrect_lemmas[error_key] += 1

                processed_sent_count += 1
                if (processed_sent_count%100)==0:
                    if is_tagging:
                        print( '{0} sentences have been lemmatized so far'.format(processed_sent_count), end='\r' )
                    else:
                        print( '{0} sentences have been evaluated so far, err_rate={1}'.format(processed_sent_count,incorrect_word_count/float(total_word_count)), end='\r' )
                
            good = True
            sent = []

            if is_tagging:
                wrt_result.write( '\n' )

        else:
            tx = line.split(u'\t')
            if len(tx)<2:
                good = False
            else:
                word = tx[1]
                lemma = tx[2]
                part_of_speech = tx[3]
                tags = tx[4].split(u'|')
                sent.append( (word,lemma,part_of_speech,tags,tx) )

    if not is_tagging:
        for (error_key,freq) in sorted(incorrect_lemmas.iteritems(), key=lambda z:-z[1]):
            word = error_key[0]
            model_lemma = error_key[1]
            corpus_lemma = error_key[2]
            part_of_speech = error_key[3]
            wrt_result.write( u'{}\t{}\t{}\t{}\t{}\n'.format(word.ljust(30), model_lemma.ljust(30), corpus_lemma.ljust(30), part_of_speech.ljust(30), freq))

    wrt_result.close()
    print( '\nDone, {0} sentences have been processed.'.format(processed_sent_count) )

    print( 'lemmatized_by_table={}'.format(lemmatized_by_table) )
    print( 'lemmatized_by_model={}'.format(lemmatized_by_model) )

    if not is_tagging:
        print( 'err_rate={0}'.format( incorrect_word_count/float(total_word_count) ) )
    
# ------------------------------------------------------------------------
else:
    print( 'Unknown scenario {0}'.format(run_mode) )
    usage()
