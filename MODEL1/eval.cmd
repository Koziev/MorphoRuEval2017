rem set CORPUS=../CORPORA/united_corpora_test.dat
set CORPUS=../Baseline/source/gikrya_test.txt
set MODEL=../GIKRYA_Models
set RESULTS=../RESULTS/model1_validation.txt
python chars_postagger_net.py eval %CORPUS% %RESULTS% %MODEL%
