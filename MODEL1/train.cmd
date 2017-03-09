rem set CORPUS=../CORPORA/united_corpora_train.dat
set CORPUS=../Baseline/source/gikrya_train.txt
set MODEL=../GIKRYA_Models
python chars_postagger_net.py learn %CORPUS% %MODEL%
