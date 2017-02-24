# -*- coding: utf-8 -*-
'''
Упаковка список специальных слов, для которых не будет выполняться частеречная разметка.
(c) Илья Козиев 2017 для morphorueval 2017 inkoziev@gmail.com
'''

from __future__ import print_function
import codecs

def add_words( src_path, part_of_speech, wrt ):
    with codecs.open( src_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            word = line.strip().lower()
            wrt.write( u'{}\t{}\n'.format(word,part_of_speech) )


output_path = u'special_words.dat'

with codecs.open( output_path, 'w', 'utf-8' ) as wrt:
    add_words( 'ADP.txt', u'ADP', wrt)
    add_words( 'CONJ.txt', u'CONJ', wrt)
    add_words( 'H.txt', u'H', wrt)
    add_words( 'PART.txt', u'PART', wrt )


