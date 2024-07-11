# Following preprocessing steps are required before classifier action
python pipe_step_4_clf_clfers/preprocess.py pipe_step_4_clf_clfers 20

echo "starting classifiers. 1. NB, 2. SVM, 3.RF"

cd pipe_step_4_clf_clfers/SVM/
python SVM.py
cd ../
cd RF/
python RF.py
cd ../
cd NB/
python NB.py
