# -*- coding: utf-8 -*-
'''
Из POSTagger-корпуса соревнования делаем корпус для обучения word2vec.
'''

from __future__ import print_function
import codecs
import collections


import sys

src_dataset = sys.argv[1]
res_dataset = 'w2v_corpus.txt'

rdr = codecs.open( src_dataset, 'r', 'utf-8' )
wrt = codecs.open( res_dataset, 'w', 'utf-8' )

sent_count=0
sent = []

for line in rdr:
    parts = line.strip().split(u'\t')
    if len(parts)==0 or len(parts[0])==0:
        # end of sentence
        sent_count += 1
        wrt.write( u'{0}\n'.format( unicode.join( u' ', sent ) ) )
        sent = []
    else:
        word = parts[1].lower()
        sent.append( word )

wrt.close()
print( 'Done, {0} sentences.'.format(sent_count) )
