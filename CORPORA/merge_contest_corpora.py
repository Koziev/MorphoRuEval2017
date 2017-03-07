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

base_src = 'gikrya_fixed.txt'
src = [ 'gikrya_fixed.txt', 'syntagrus_full_fixed.ud', 'RNCgoldInUD_Morpho.conll', 'unamb_sent_14_6.conllu' ]

result = 'united_corpora.dat'
result_train = 'united_corpora_train.dat'
result_test = 'united_corpora_test.dat'

test_share = 0.1 # доля тестового набора при делении на train и test

# --------------------------------------------------------------------------------

word2lemmas = {u'ней': u'он', u'ним': u'он', u'ними': u'он', u'она': u'он', u'оно': u'он',
                u'они': u'он', u'него': u'он', u'нее': u'он', u'них': u'он', u'более': u'много',
               u'нему': u'он', u'им': u'он', u'ему': u'он', u'т.е.': u'т.е.'}

pos_word2lemmas = {u'PRON:его': u'он', u'ADP:со': u'со', u'ADP:ко': u'ко', u'ADP:подо': u'подо',
                   u'ADP:ото': u'ото', u'ADP:во': u'во',
                   u'ADP:надо': u'надо', u'ADV:больше': u'много', u'PRON:нем': u'он',
                   u'PRON:ей': u'он', u'PRON:ее': u'он', u'PRON:их': u'он', u'PRON:его': u'он',
                   u'PRON:это': u'этот', u'PRON:то': u'тот', u'PRON:чего': u'что', u'PRON:том': u'тот',
                   u'PRON:тем': u'тот', u'PRON:того': u'тот', u'PRON:тому': u'тот', u'PRON:той': u'тот',  u'PRON:тем': u'тот',
                   u'DET:это': u'этот', u'DET:тот': u'тот', u'DET:все': u'весь',
                   u'DET:многие': u'многие', u'DET:многих': u'многие', u'DET:многим': u'многие',
                   u'ADV:менее': u'мало', u'NUM:менее': u'мало', u'DET:то': u'тот', u'ADV:лучше': u'хорошо',
                   u'PRON:ничего': u'ничто', u'NOUN:людей': u'человек', u'DET:какие': u'какой',
                   u'NUM:больше': u'много', u'ADJ:должна': u'должен', u'ADJ:должны': u'должен', u'ADJ:должно': u'должен',
                   u'ADV:дальше': u'далеко', u'ADJ:готов': u'готовый', u'ADV:раньше': u'рано', u'NOUN:деньги': u'деньги',
                   u'ADJ:должен': u'должен', u'ADV:чаще': u'часто', u'NUM:двух': u'два', u'NOUN:матери': u'мать',
                   u'NOUN:дети': u'ребенок', u'ADV:позже': u'поздно', u'NUM:четырех': u'четыре', u'NUM:двух': u'два',
                   u'ADV:хуже': u'плохо', u'PRON:что': u'что', u'ADV:скорее': u'скоро', u'NUM:трех': u'три',
                   u'DET:каких': u'какой', u'PRON:всех': u'все', u'ADV:выше': u'высоко', u'PRON:ими': u'он',
                   u'ADJ:второе': u'второй', u'ADV:ниже': u'низко', u'ADV:далее': u'далеко', u'ADV:меньше': u'мало',
                   u'PRON:нею': u'он', u'NOUN:россии': u'россии', u'NOUN:россию': u'россию', u'NOUN:россией': u'россия',
                   u'NOUN:россии': u'россия', u'NOUN:москве': u'москва', u'NOUN:японии': u'япония', u'NOUN:москвы': u'москва',
                   u'NOUN:германии': u'германия', u'NOUN:италии': u'италия', u'NOUN:абхазии': u'абхазия',
                   u'NOUN:украины': u'украина', u'NOUN:великобритании': u'великобритания', u'NOUN:украине': u'украина',
                   u'NOUN:африка': u'африка', u'NOUN:турции': u'турция', u'NOUN:лет': u'год', u'NUM:сколько': u'сколько',
                   u'PRON:ею': u'он', u'NOUN:китая': u'китай', u'ADJ:одно': u'один'
                  }

def correct_lemma( word, part_of_speech, lemma, tags_str ):
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

print( 'Merging corpora...' ) 
# Теперь собираем все в единый новый корпус
wrt = codecs.open( result, 'w', 'utf-8')
wrt_train = codecs.open( result_train, 'w', 'utf-8')
wrt_test = codecs.open( result_test, 'w', 'utf-8')

total_words = 0
total_sents = 0

for corp_path in src:
    print( 'Processing {}...'.format(corp_path) )
    with codecs.open( corp_path, 'r', 'utf-8') as rdr:
        token_num=1
        sent = []
        corpus_line_num = 0

        for line0 in rdr:
            line = line0
            corpus_line_num += 1
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
                total_sents += 1
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

                    loword = word.lower().replace(u'ё',u'е')

                    if pos==u'PROPN':
                        pos = u'NOUN'

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

                    if pos==u'VERB' and u'Aspect=' in new_tags_str:
                        new_tags_str = new_tags_str.replace(u'Aspect=Imp|', u'')
                        new_tags_str = new_tags_str.replace(u'Aspect=Perf|', u'')

                    if loword!=u'нет' and loword!=u'нету' and pos==u'VERB' and u'VerbForm=Fin' in new_tags_str and u'Voice=' not in new_tags_str:
                        # в корпусе ГИКРИЯ глаголы в личной форме имеют тэг Voice=Act или Voice=Mid
                        if loword.endswith(u'ся') or loword.endswith(u'сь'):
                            new_tags_str = new_tags_str + u'|Voice=Mid'
                        else:
                            new_tags_str = new_tags_str + u'|Voice=Act'

                    # в корпусе unamb_sent_14_6.conllu краткие формы прилагательного имеют тег Case=Nom
                    # и не имеют тега Degree=Pos. Поправим это.
                    if pos==u'ADJ' and u'Variant=Short' in new_tags_str:
                        if u'Case=Nom|' in new_tags_str:
                            new_tags_str = new_tags_str.replace(u'Case=Nom|',u'')
                        if u'Degree=Pos' not in new_tags_str:
                            new_tags_str = u'Degree=Pos|' + new_tags_str

                    lemma = correct_lemma(word, pos, lemma0, new_tags_str)
                        
                    wrt.write( u'{}\t{}\t{}\t{}\t{}\n'.format(itoken,word,lemma,pos,new_tags_str) )
                    
                    if len(word)==0 or len(lemma)==0 or len(pos)==0:
                        # В корпусе RNCgoldInUD_Morpho.conll есть невероятные токены - с полностью пустой
                        # цепочкой символов и леммой. Игнорируем их.
                        print('Invalid token data: token num={} line num={}'.format(itoken,corpus_line_num) )
                    else:    
                        sent.append( (itoken,word,lemma,pos,new_tags_str) )
                        total_words += 1

wrt.close()
wrt_train.close()
wrt_test.close()

print( 'Done, total_words={} total_sents={}'.format(total_words,total_sents) ) 
