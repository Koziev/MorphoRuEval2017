set CORPUS=../Baseline/source/gikrya_new_test.out
set RESULTS=../RESULTS/ensemble_validation.txt
set MODEL=../GIKRYA_Models
python apply_ensemble.py eval %CORPUS% %RESULTS% %MODEL%
pause
