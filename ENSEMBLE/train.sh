#CORPUS=../CORPORA/united_corpora_train.dat
CORPUS=../Baseline/source/gikrya_train.txt
MODEL=../GIKRYA_Models

python train_memtable.py $CORPUS $MODEL
read -rsp $'memtable training has completed, press enter to continue...\n'

python train_NLTK.py $CORPUS $MODEL
read -rsp $'NLTK training has completed, press enter to continue...\n'

python 'train_chars_postagger(2).py' $CORPUS $MODEL
read -rsp $'chars_postagger training has completed, press enter to continue...\n'

python 'train_decisiontrees(2).py' $CORPUS $MODEL
read -rsp $'decisiontrees training has completed, press enter to continue...\n'
