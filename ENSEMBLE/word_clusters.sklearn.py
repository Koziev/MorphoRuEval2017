# -*- coding: utf-8 -*-
'''
Выполняем кластеризацию w2v векторов слов с помощью scikit-learn K-means.

w2v модель нужно предварительно создать с помощью word2vec или gensim.

Визуализация получающихся кластеров выполняется через
вывод ближайшего к центроиду слова и вывод всех слов в кластере
с сортировкой по убыванию их частоты. Частоту слов берем из анализа
текстового корпуса.
'''
from __future__ import print_function
import gensim
import sklearn.cluster
import scipy.spatial.distance
import numpy as np
import codecs
import collections
import math
import sys


# Бинарный файл word2vec модели
w2v_path = 'w2v_corpus.model'

# путь к текстовому файлу с корпусом предложений, с помощью которого
# получаются частоты слов.
corp_path = 'w2v_corpus.txt'

NWORDS = 1000000
n_cluster = 1000

# Путь к текстовому файлу, куда мы сохраним визуализацию кластеров.
res_path = 'clusters.txt'

# Путь к текстовому файлу, куда мы сохраним привязку слов к кластерам
res2_path = 'word2cluster.dat'

# ----------------------------------------------------------------------

print( 'Counting word occurencies in {0}...'.format(corp_path) )
word2freq = collections.defaultdict( lambda:0 )
nline=0
with codecs.open(corp_path, "r", "utf-8") as rdr:
    for line in rdr:
        nline += 1
        for word in line.strip().split(' '):
            word2freq[word] += 1
print( 'Finished, {0} lines, {1} unique words'.format(nline,len(word2freq)) )


print( 'Loading w2v model...' )
w2v = gensim.models.Word2Vec.load_word2vec_format(w2v_path, binary=True)

nword = len(w2v.vocab)
print( 'Number of words={0}'.format( nword ) )
vec_len = len( w2v.syn0[0] )
print( 'Vector length={0}'.format(vec_len) )

# берем NWORDS самых частотных слов
words = sorted( [ w for w in w2v.vocab if w in word2freq ], key=lambda z:-word2freq[z] )[:NWORDS]

# сохраним этот список слов в файле для визуального контроля
with codecs.open( 'most_frequent_words.txt', 'w', 'utf-8' ) as wrt:
    for word in words:
        wrt.write( u'{0}\t{1}\n'.format(word,word2freq[word]))

print( 'Filling the matrix of wordvectors...' )
nrow = len(words)
X = np.zeros( (nrow,vec_len) )
for (i,word) in enumerate(words):
    X[i] = w2v[word]
print( 'Done, {0} words accepted.'.format(len(words)) )

# ----------------------------------------------------------------------

print( 'Start k-means for {0} vectors, {1} clusters...'.format( len(words), n_cluster ) )
kmeans = sklearn.cluster.KMeans( n_clusters=n_cluster, max_iter=10, verbose=1, copy_x=False, n_jobs=1, algorithm='auto')
kmeans.fit(X)
print( 'Finished.' )

codebook = kmeans.cluster_centers_
labels0 = kmeans.labels_

# ----------------------------------------------------------------------

# Теперь распределим весь лексикон по кластерам, чтобы для **каждого**
# слова знать номер его кластера.

words = sorted( [ w for w in w2v.vocab if w in word2freq ], key=lambda z:-word2freq[z] )

print( 'Labeling all {0} words in lexicon...'.format( len(words) ) )

batch_size = 1024
remaining_count = len(words)
word_index = 0

with codecs.open( res2_path, "w", "utf-8") as wrt:
    while remaining_count>0:

        nrow = min( batch_size, remaining_count )
        X = np.zeros( (nrow,vec_len) )
        for i in range(nrow):
            X[i] = w2v[ words[word_index+i] ]

        labels = kmeans.predict(X)

        for (i,icluster) in enumerate(labels):
            wrt.write( u'{0}\t{1}\n'.format( words[word_index+i], icluster ) )

        remaining_count -= nrow
        word_index += nrow
        print( '{0}/{1}'.format( word_index, len(words) ), end='\r' )

print( '\nAll done.' )


# ----------------------------------------------------------------------

# кластеры отсортируем в порядке убывания частоты слов
print( u'Printing human-readable results to {0}'.format(res_path) )
cluster_contents = []

for i in range(n_cluster):
    print( '{0}/{1}'.format(i,n_cluster), end='\r' )
    sys.stdout.flush()
    
    words_in_cluster = [ words[iword] for (iword,l) in enumerate(labels0) if l==i ]
    words_in_cluster = sorted( words_in_cluster, key=lambda w:-word2freq[w]*scipy.spatial.distance.cosine(codebook[i],w2v[w]) )
    words_in_cluster = words_in_cluster[: min(50,len(words_in_cluster))]
    
    # суммарная частота топовых слов в кластере
    words_sum_freq = sum( ( word2freq[w] for w in words_in_cluster ) )
    
    # в распечатке покажем слово, ближайшее к центру кластера
    cluster_word = w2v.most_similar( positive=[codebook[i]], topn=1 )[0]
    
    # а также самое частотное слово в кластере
    topfreq_word = words_in_cluster[0]
    
    s = u'cluster #{0} nearest:{1} {2} top_frequent:{3} ==>\n'.format(i,cluster_word[0],cluster_word[1],topfreq_word) + unicode.join( u' ', [ w for w in words_in_cluster ] )
    cluster_contents.append( (words_sum_freq,s,topfreq_word) )


with codecs.open( 'cluster_topword.txt', 'w', 'utf-8' ) as wrt:
    for (i,cluster) in enumerate(cluster_contents):
        wrt.write( u'{0}\t{1}\n'.format(i,cluster[2]) )

with codecs.open( res_path, "w", "utf-8") as wrt:
    for x in sorted( cluster_contents, key=lambda z:-z[0] ):
        wrt.write( x[1] + u'\n\n' )

# ----------------------------------------------------------------------

