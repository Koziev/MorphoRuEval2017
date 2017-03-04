# -*- coding: utf-8 -*-
'''
(c) Илья Козиев 2017 для morphorueval 2017 inkoziev@gmail.com
Слияние нескольких обучающих корпусов, предоставленных организаторами конкурса,
в один корпус с дополнительными правками.
Формируется один общий корпус, а также отдельно обучающая и тестовая порции.
'''

from __future__ import print_function
import codecs
import random

base_src = 'GIKRYA_texts.txt'
src = [ 'GIKRYA_texts.txt', 'RNCgoldInUD_Morpho.conll', 'unamb_sent_14_6.conllu' ]

result = 'united_corpora.dat'
result_train = 'united_corpora_train.dat'
result_test = 'united_corpora_test.dat'

test_share = 0.1 # доля тестового набора при делении на train и test

# --------------------------------------------------------------------------------

word2lemmas = {u'ней': u'он', u'ним': u'он', u'ними': u'он', u'она': u'он', u'оно': u'он',
                u'они': u'он', u'него': u'он', u'нее': u'он', u'них': u'он', u'более': u'много',
               u'нему': u'он', u'им': u'он', u'ему': u'он', u'т.е.': u'т.е.'}

pos_word2lemmas = {u'PRON:его': u'он', u'ADP:со': u'со', u'ADP:ко': u'ко', u'ADP:подо': u'подо', u'ADP:ото': u'ото',
                   u'ADP:надо': u'надо', u'ADV:больше': u'много', u'PRON:нем': u'он',
                   u'PRON:ей': u'он', u'PRON:ее': u'он', u'PRON:их': u'он', u'PRON:его': u'он',
                   u'PRON:это': u'этот', u'PRON:то': u'тот', u'PRON:чего': u'что', u'PRON:том': u'тот',
                   u'PRON:тем': u'тот', u'PRON:того': u'тот',
                   u'DET:это': u'этот', u'DET:тот': u'тот', u'DET:все': u'весь',
                   u'DET:многие': u'многие', u'DET:многих': u'многие', u'DET:многим': u'многие',
                   u'ADV:менее': u'мало'
                  }

def correct_lemma( word, part_of_speech, lemma ):
    uword = word.lower().replace(u'ё',u'е')
    if uword in word2lemmas:
        return word2lemmas[uword]
    else:
        k = part_of_speech+u':'+uword
        if k in pos_word2lemmas:
            return pos_word2lemmas[k]
        return lemma

# --------------------------------------------------------------------------------



# наборы тегов во всех корпусах разные, поэтому попробуем отсечь ненужные теги.
# для этого построим по каждой части речи список тегов, встречающихся в базовом корпусе.
pos2tags = dict()
with codecs.open( base_src, 'r', 'utf-8' ) as rdr:
    for line0 in rdr:
        line = line0.strip()
        px = line.split(u'\t')
        if len(px)==5:
            pos = px[3]
            tags_str = px[4]
            tags = [] if tags_str==u'_' else tags_str.split(u'|')
            if pos not in pos2tags:
                pos2tags[pos] = set()
            pos2tags[pos].update( tags )


# Теперь собираем все в единый новый корпус
wrt = codecs.open( result, 'w', 'utf-8')
wrt_train = codecs.open( result_train, 'w', 'utf-8')
wrt_test = codecs.open( result_test, 'w', 'utf-8')

for corp_path in src:
    with codecs.open( corp_path, 'r', 'utf-8') as rdr:
        token_num=1
        sent = []

        for line0 in rdr:
            line = line0
            if len(line.strip())==0:
                wrt.write( u'\n' )
                wrt.flush()

                wrt1 = wrt_test if random.uniform(0.0,1.0)<test_share else wrt_train

                for (itoken,word,lemma,pos,new_tags_str) in sent:
                    wrt1.write(u'{}\t{}\t{}\t{}\t{}\n'.format(itoken, word, lemma, pos, new_tags_str))

                wrt1.write(u'\n')
                wrt1.flush()

                token_num=1
                sent = []
            elif line[0]==u'=':
                continue
            else:
                if line[0]==u'\t': # 'RNCgoldInUD_Morpho.conll'
                    line = str(token_num)+line

                px = line.split(u'\t')
                if len(px)>=5:
                    token_num += 1
                    itoken = px[0]
                    word = px[1]
                    lemma0 = px[2]
                    pos = px[3]

                    lemma = correct_lemma(word, pos, lemma0)

                    tags_str = px[4].strip()
                    if tags_str==u'_':
                        tags_str = u''

                    if len(tags_str)==0 and len(px)>5 and px[5]!=u'_':
                        tags_str = px[5]

                    new_tags_str = u''
                    if pos in pos2tags:
                        new_tags_str = unicode.join( u'|', [ tag for tag in tags_str.split(u'|') if tag in pos2tags[pos] ] )
                    if len(new_tags_str)==0:
                        new_tags_str=u'_'
                    wrt.write( u'{}\t{}\t{}\t{}\t{}\n'.format(itoken,word,lemma,pos,new_tags_str) )
                    sent.append( (itoken,word,lemma,pos,new_tags_str) )

wrt.close()
wrt_train.close()
wrt_test.close()
